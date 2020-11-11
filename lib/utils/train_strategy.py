__all__ = ["scale_lr"]


def scale_lr(optimizer, k):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] *= k
    print('lr multiply {}'.format(k))
