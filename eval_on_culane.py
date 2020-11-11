import os
import glob
import logging

import cv2
import torch
import numpy as np
from tqdm.autonotebook import tqdm

import lib.network
import lib.dataset
import lib.utils

# PyTorch基础设置
torch.manual_seed(1)  # 设置随机种子
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


# 定义训练流程
class PipeLine(object):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.save_root = "{}/checkpoints/{}/lanes4test".format(self.cfg.save_root, self.cfg.exp_name)
        if not os.path.exists(os.path.join(self.save_root, "evaluate_result")):
            os.makedirs(os.path.join(self.save_root, "evaluate_result"))

        # 初始化DataLoader
        self.ds_type = lib.dataset.__dict__["CULaneTest"]
        self.test_loader = lib.utils.DataLoaderX(self.ds_type(self.cfg.data_root),
                                                 batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers,
                                                 shuffle=False, pin_memory=True, drop_last=False)

        # 初始化模型
        self.net = lib.network.model.__dict__[self.cfg.model_type](num_classes=self.ds_type.num_class).cuda()
        logging.info('Model Type: {}, Total Params: {:.2f}M'.format(
            self.cfg.model_type, sum(p.numel() for p in self.net.parameters())/1e6)
        )

        # 加载模型
        self.load_checkpoint()

    def run(self):
        self.net.eval()
        progress_bar = tqdm(self.test_loader)
        with torch.no_grad():
            for data in progress_bar:
                img, img_path = data
                img = img.cuda(non_blocking=True)
                outputs = self.net(img)
                seg_maps = outputs["seg_maps"].cpu()
                exist_codes = outputs["exist_codes"].cpu()
                seg_maps = torch.softmax(seg_maps, dim=1)

                for b in range(seg_maps.shape[0]):
                    seg_map = seg_maps[b]
                    exist_code = [1 if exist_codes[b, i] > 0.5 else 0 for i in range(4)]
                    lane_xys = self.prob2lines(seg_map, exist_code,
                                               smooth=True, thresh=0.5,
                                               resize_shape=(590, 1640), y_px_gap=20, pts=18)

                    tmp_path = img_path[b].split(self.cfg.data_root)[1]
                    save_dir = self.save_root + os.path.dirname(tmp_path)
                    save_name = os.path.basename(tmp_path)[:-3] + "lines.txt"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    with open(os.path.join(save_dir, save_name), "w") as f:
                        for lane in lane_xys:
                            for x, y in lane:
                                print("{} {}".format(x, y), end=" ", file=f)
                            print(file=f)

                progress_bar.update(1)
        progress_bar.close()

        # evaluate
        os.system("sh ./eval/CULane/Run.sh " + self.cfg.exp_name)
        outputs = glob.glob(os.path.join(self.save_root, "evaluate_result") + "/out*.txt")

        tp, fp, fn = 0, 0, 0
        for output in outputs:
            info = self.read_helper(output)
            tp += int(info["tp"])
            fp += int(info["fp"])
            fn += int(info["fn"])

        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)

        with open(os.path.join(self.save_root, "evaluate_result", "f1.txt"), "w") as f:
            f.write("f1-Score: {:.3f}".format(f1))
        logging.info("| F1-Score: {:.3f} |".format(f1))

    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.cfg.save_root, "checkpoints", self.cfg.exp_name, "best.pt")
        if os.path.isfile(checkpoint_path):
            logging.info("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            lib.utils.load_model(self.net, checkpoint_path)
            logging.info("=> loaded checkpoint '{}' (epoch {})\n".format(checkpoint_path, checkpoint['epoch']))
        else:
            logging.error("=> no checkpoint @ '{}'\n".format(checkpoint_path))

    @staticmethod
    def read_helper(path):
        lines = open(path, 'r').readlines()[1][:-1]
        values = lines.split(' ')[1::2]
        keys = lines.split(' ')[0::2]
        keys = [key[:-1] for key in keys]
        res = {k: v for k, v in zip(keys, values)}
        return res

    def prob2lines(self, seg_map, exist_code, resize_shape=None, smooth=True, y_px_gap=20, pts=None, thresh=0.5):
        """
        Arguments:
        ----------
        seg_pred: np.array size (5, h, w)
        resize_shape:  reshape size target, (H, W)
        exist:   list of existence, e.g. [0, 1, 1, 0]
        smooth:  whether to smooth the probability or not
        y_px_gap: y pixel gap for sampling
        pts:     how many points for one lane
        thresh:  probability threshold
        Return:
        ----------
        coordinates: [x, y] list of lanes, e.g.: [ [[9, 569], [50, 549]] ,[[630, 569], [647, 549]] ]
        """
        if resize_shape is None:
            resize_shape = seg_map.shape[1:]  # seg_map (5, h, w)
        _, h, w = seg_map.shape
        rh, rw = resize_shape
        coordinates = []

        if pts is None:
            pts = round(rh / 2 / y_px_gap)

        seg_map = np.ascontiguousarray(np.transpose(seg_map, (1, 2, 0)))
        for i in range(4):
            prob_map = seg_map[..., i + 1]
            if smooth:
                prob_map = cv2.blur(prob_map, (9, 9), borderType=cv2.BORDER_REPLICATE)
            if exist_code[i] > 0:
                coords = self.get_lane(prob_map, y_px_gap, pts, thresh, resize_shape)
                coordinates.append([[coords[j], rh - 1 - j * y_px_gap] for j in range(pts) if coords[j] > 0])

        return coordinates

    @staticmethod
    def get_lane(prob_map, y_px_gap, pts, thresh, resize_shape=None):
        """
        Arguments:
        ----------
        prob_map: prob map for single lane, np array size (h, w)
        resize_shape:  reshape size target, (H, W)
        Return:
        ----------
        coords: x coords bottom up every y_px_gap px, 0 for non-exist, in resized shape
        """
        if resize_shape is None:
            resize_shape = prob_map.shape
        h, w = prob_map.shape
        rh, rw = resize_shape

        coords = np.zeros(pts)
        for i in range(pts):
            y = int(h - i * y_px_gap / rh * h - 1)
            if y < 0:
                break
            line = prob_map[y, :]
            id = np.argmax(line)
            if line[id] > thresh:
                coords[i] = int(id / w * rw)
        if (coords > 0).sum() < 2:
            coords = np.zeros(pts)

        return coords


if __name__ == '__main__':
    import argparse

    # 设置日志消息格式
    logging.basicConfig(format="%(asctime)s-%(message)s", level=logging.INFO)

    # 读取运行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()

    # 加载配置文件
    cfg = lib.utils.load_yaml_config(args.config_path)

    # 实例化评估流程并开始评估
    pipeline = PipeLine(cfg)
    pipeline.run()
