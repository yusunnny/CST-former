import math

def adjust_learning_rate(optimizer, epoch, params):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < params['warmup_epochs']:
        lr = params['lr'] * epoch / params['warmup_epochs']
    else:
        lr = params['min_lr'] + (params['lr'] - params['min_lr']) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - params['warmup_epochs']) / (params['nb_epochs'] - params['warmup_epochs'])))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def adjust_learning_rate_ramp(optimizer, epoch, params):
    """Adjust learning rate with ramp and maintain"""
    if epoch < params['warmup_epochs']:
        lr = params['lr'] * epoch / params['warmup_epochs']
    else:
        lr = optimizer.param_groups[0]["lr"]

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def adjust_lr_by_epoch(optimizer, epoch, params):
    """Decay the learning rate with ramp,
    maintaining the highest lr for a while, and then half-cycle cosine """
    stay_epoch = params['lr_by_epoch_stay_epoch']
    warm_up1 = params['warmup_epochs']

    if epoch < warm_up1:
        lr = params['lr'] * epoch / warm_up1
    elif epoch < stay_epoch:
        lr = params['lr']
    else:
        lr = params['min_lr'] + (params['lr'] - params['min_lr']) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - stay_epoch) / (params['nb_epochs'] - stay_epoch)))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
