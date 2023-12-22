import torch
from tqdm import tqdm
import utility.lr_sched as lr_sched
import matplotlib.pyplot as plot
plot.switch_backend('agg')

from augmentation.data_aug_module import data_augment_module

def train_epoch(data_generator, optimizer, model, criterion, params, device, epoch_cnt):
    nb_train_batches, train_loss = 0, 0.
    model.train()

    total_batches = data_generator.get_total_batches_in_data()
    if params["augment"]:
        augment_module = data_augment_module(params)

    with tqdm(total=total_batches) as pbar:
        for data, target in data_generator.generate():

            # learning rate scheduler
            if params['lr_scheduler']:
                if params['lr_by_epoch']:
                    lr_sched.adjust_lr_by_epoch(optimizer, nb_train_batches / total_batches + epoch_cnt, params)
                elif params['lr_ramp']:
                    lr_sched.adjust_learning_rate_ramp(optimizer, nb_train_batches / total_batches + epoch_cnt, params)
                else:
                    lr_sched.adjust_learning_rate(optimizer, nb_train_batches / total_batches + epoch_cnt, params)

            if params["augment"]:
                if not params["multi_accdoa"]:
                    data, target = augment_module.aug_and_mix(data, target)
                else:
                    data, target = augment_module.aug_and_mix_multi(data, target)
                data, target = data.to(device).float(), target.to(device).float()
            else:
                data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()

            optimizer.zero_grad()
            output = model(data.contiguous())

            # process the batch of data based on chosen mode
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            nb_train_batches += 1
            if params['quick_test'] and nb_train_batches == 4:
                break
            pbar.update(1)
        train_loss /= nb_train_batches

    return train_loss, optimizer.param_groups[0]["lr"]
