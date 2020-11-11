import os
import yaml
import logging
import torch
import shutil
from prefetch_generator import BackgroundGenerator

__all__ = ["save_checkpoint", "load_yaml_config", "load_model", "DataLoaderX"]


class AttrDict(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, **kwargs):
        super(AttrDict, self).__init__()
        super(AttrDict, self).update(kwargs)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))


def load_yaml_config(fp):
    # 读取 yaml file
    with open(fp, 'r') as f:
        fd = f.read()
    cfg_dict = yaml.safe_load(fd)

    # 实例化配置参数
    cfg = AttrDict(**cfg_dict)

    # 打印实例化结果
    logging.info(" <{}> : {}".format(os.path.basename(fp), cfg))
    return cfg


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict_ = checkpoint['state_dict']
    try:
        logging.debug('=>> load model (epoch: {}, acc: {:.3f})'.format(checkpoint['epoch'], checkpoint['current_acc']))
    except:
        pass
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = ''
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                logging.debug('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            logging.debug('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            logging.debug('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    return model


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def save_checkpoint(state, is_best, save_root, filename='last.pt'):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    fp = os.path.join(save_root, filename)
    torch.save(state, fp)

    if is_best:
        shutil.copyfile(fp, os.path.join(save_root, 'best.pt'))


if __name__ == "__main__":
    yaml_pth = "../../exp_configs/resnet18_CigaOpenOrNot_exp1.yml"
    load_yaml_config(yaml_pth)
