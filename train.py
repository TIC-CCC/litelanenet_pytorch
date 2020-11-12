import os
import logging

import torch
from torch.utils.tensorboard import SummaryWriter
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

        # 初始化checkpoint路径
        self.save_root = '{}/checkpoints/{}'.format(self.cfg.save_root, self.cfg.exp_name)
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)

        # 初始化tensor_board logger路径
        self.tb_logger = SummaryWriter(self.save_root + '/tb_log')

        # 初始化DataLoader
        self.ds_type = lib.dataset.__dict__[self.cfg.dataset_type]
        self.train_loader = lib.utils.DataLoaderX(self.ds_type(self.cfg.data_root, is_train=True),
                                                  batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers,
                                                  shuffle=True, pin_memory=True, drop_last=True)
        self.val_loader = lib.utils.DataLoaderX(self.ds_type(self.cfg.data_root, is_train=False),
                                                batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers,
                                                shuffle=True, pin_memory=True, drop_last=False)

        # 初始化模型
        self.net = lib.network.model.__dict__[self.cfg.model_type]().cuda()
        logging.info('Model Type: {}, Total Params: {:.2f}M'.format(
            self.cfg.model_type, sum(p.numel() for p in self.net.parameters())/1e6)
        )
        if self.cfg.pretrain_model is not None:
            self.net = lib.utils.load_model(self.net, self.cfg.pretrain_model)

        # 初始化损失函数
        self.loss_fn = lib.network.loss.__dict__[self.cfg.loss_type]()

        # 初始化优化器
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.cfg.lr,
                                         momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)

        # 初始化训练参数
        self.start_epoch, self.best_epoch, self.step, self.best_acc, self.current_acc = 5 * [0]

        # 加载预训练模型，覆盖模型参数，优化器参数，训练参数
        self.load_checkpoint()

    def run(self):
        for epoch in range(self.start_epoch + 1, self.cfg.epochs + 1):
            if epoch in self.cfg.lr_decay_steps:
                lib.utils.scale_lr(self.optimizer, 0.1)  # 学习率下降

            logging.info('| epoch:{}/{}, last-acc({}):{:.3f}, best-acc({}):{:.3f}|'.format(
                epoch, self.cfg.epochs, epoch-1, self.current_acc, self.best_epoch, self.best_acc))

            # 训练一个周期
            self.train_one_epoch()

            if epoch % self.cfg.check_interval == 0:
                # 验证一个周期
                self.val_one_epoch()
                self.tb_logger.add_scalar('info/val-acc', self.current_acc, epoch)

                # 判断是否更新性能最好的模型(best.pt)
                is_best = self.current_acc > self.best_acc
                if is_best:
                    self.best_epoch = epoch
                    self.best_acc = self.current_acc
                lib.utils.save_checkpoint({'epoch': epoch, 'step': self.step,
                                           'state_dict': self.net.state_dict(),
                                           'optimizer': self.optimizer.state_dict(),
                                           'current_acc': self.current_acc,
                                           'best_acc': self.best_acc, 'best_epoch': self.best_epoch}
                                          , is_best, save_root=self.save_root)

    def train_one_epoch(self):
        self.net.train()
        progress_bar = tqdm(self.train_loader)
        for data in progress_bar:
            img, seg_lbl, exist_lbl = [v.cuda(non_blocking=True) for v in data]

            # forward propagation
            outputs = self.net(img)

            # compute losses
            losses = self.loss_fn(outputs, [seg_lbl, exist_lbl])

            # back propagation
            self.optimizer.zero_grad()
            losses["loss"].backward()
            self.optimizer.step()

            # 记录训练信息
            progress_bar.set_description('TRAIN -> Step: {}. lr: {:.8f}. Loss: {:.5f}.'.
                                         format(self.step,
                                                self.optimizer.param_groups[-1]['lr'],
                                                losses["loss"].item()))
            if self.step % 500 == 0:
                self.tb_logger.add_scalar('info/loss', losses["loss"], self.step)
                self.tb_logger.add_scalar('info/focal_loss', losses["focal_loss"], self.step)
                self.tb_logger.add_scalar('info/iou_loss', losses["iou_loss"], self.step)
                self.tb_logger.add_scalar('info/bce_loss', losses["bce_loss"], self.step)
                self.tb_logger.add_scalar('info/lr', self.optimizer.param_groups[-1]['lr'], self.step)

            # 更新迭代次数
            self.step += 1

        progress_bar.close()

    def val_one_epoch(self):
        self.net.eval()
        intersections = torch.zeros(self.ds_type.num_class)
        unions = torch.zeros(self.ds_type.num_class)
        miou = 0

        progress_bar = tqdm(self.val_loader)
        with torch.no_grad():
            for data in progress_bar:
                img, seg_lbl, _ = data
                img = img.cuda(non_blocking=True)
                outputs = self.net(img)
                seg_maps = outputs["seg_maps"].cpu()
                seg_maps = torch.argmax(seg_maps, dim=1)

                for i in range(self.ds_type.num_class):
                    t_seg = (seg_maps == i)
                    t_label = (seg_lbl == i)
                    intersection = t_seg * t_label
                    union = t_seg + t_label
                    intersections[i] += torch.sum(intersection)
                    unions[i] += torch.sum(union)

                ious = [i / (u + 1e-8) for i, u in zip(intersections.numpy(), unions.numpy())]
                miou = float(np.mean(ious))
                progress_bar.set_description('VALID -> Acc: {:.3f}'.format(miou))

        progress_bar.close()
        self.current_acc = miou

    def load_checkpoint(self):
        if self.cfg.checkpoint:
            checkpoint_path = self.cfg.checkpoint
            if os.path.isfile(checkpoint_path):
                logging.info("=> loading checkpoint '{}'".format(checkpoint_path))
                checkpoint = torch.load(checkpoint_path)
                self.start_epoch = checkpoint['epoch']
                self.current_acc = checkpoint['current_acc']
                self.best_acc = checkpoint['best_acc']
                self.step = checkpoint['step']
                self.best_epoch = checkpoint['best_epoch']
                lib.utils.load_model(self.net, checkpoint_path)
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                logging.info("=> loaded checkpoint '{}' (epoch {})\n".format(checkpoint_path, checkpoint['epoch']))
            else:
                raise "=> no checkpoint found at '{}\n'".format(checkpoint_path)


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

    # 实例化训练流程并开始训练
    pipeline = PipeLine(cfg)
    pipeline.run()
