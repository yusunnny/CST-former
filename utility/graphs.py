import os
import numpy as np
import matplotlib.pyplot as plt

def draw_loss(log_dir, epoch, best_val_epoch,learning_rate,
                  tr_loss, val_loss, val_seld_scr, val_er, val_f, val_le, val_lr,
                    train_loss, valid_loss, valid_seld_scr, valid_ER, valid_F, valid_LE, valid_LR, learning_rate_rec):
    # -------------------------------
    # Draw Loss Graph
    # -------------------------------
    epoch_axis = np.arange(epoch + 1)
    train_loss[epoch] = tr_loss
    valid_loss[epoch] = val_loss
    valid_ER[epoch] = val_er
    valid_F[epoch] = val_f
    valid_LE[epoch] = val_le/180
    valid_LR[epoch] = val_lr
    valid_seld_scr[epoch] = val_seld_scr
    learning_rate_rec[epoch] = learning_rate

    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(10, 16))
    plt.subplot(3, 1, 1)
    plt.plot(epoch_axis, train_loss[:epoch + 1], 'r', label='Train')
    plt.plot(epoch_axis, valid_loss[:epoch + 1], 'b', label='Valid')
    plt.axvline(x=best_val_epoch, color='k', linestyle='--', label='best val epoch')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.ylim((1e-4, 0.055))
    plt.legend(fontsize=12)
    plt.title('Losses')

    plt.subplot(3,1,2)
    # plt.plot(epoch_axis, train_seld_scr[:epoch + 1], 'r', label='Train')
    plt.plot(epoch_axis, valid_seld_scr[:epoch + 1], 'k', label='Seld score')
    plt.plot(epoch_axis, valid_ER[:epoch + 1], 'b', label='Error Rate')
    plt.plot(epoch_axis, valid_F[:epoch + 1], 'g', label='F-score')
    plt.plot(epoch_axis, valid_LE[:epoch + 1], 'orange', label='Localization Error')
    plt.plot(epoch_axis, valid_LR[:epoch + 1], 'r', label='Localization Recall')
    plt.axvline(x=best_val_epoch, color='k', linestyle='--', label='best val epoch')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('SELD metrics')
    # plt.yscale('log')
    plt.ylim((0, 1))
    plt.legend(fontsize=12)
    plt.title('SELD metrics on validation dataset')

    plt.subplot(3,1,3)
    plt.plot(epoch_axis, learning_rate_rec[:epoch + 1], 'g')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate')

    plt.tight_layout()
    fig_name = '{}/{}'.format(log_dir, 'loss_graph.png')
    plt.savefig(fig_name, dpi=150)
    plt.close()

    return train_loss, valid_loss, valid_seld_scr, valid_ER, valid_F, valid_LE, valid_LR, learning_rate_rec


def draw_loss_graph_baseline(log_dir, epoch, best_val_epoch, tr_loss, val_loss, val_seld_scr,
                    train_loss, valid_loss, valid_seld_scr):
    # -------------------------------
    # Draw Loss Graph
    # -------------------------------
    epoch_axis = np.arange(epoch + 1)
    train_loss[epoch] = tr_loss
    valid_loss[epoch] = val_loss
    # train_seld_scr[epoch] = tr_seld_scr
    valid_seld_scr[epoch] = val_seld_scr

    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(10, 16))
    plt.subplot(2, 1, 1)
    plt.plot(epoch_axis, train_loss[:epoch + 1], 'r', label='Train')
    plt.plot(epoch_axis, valid_loss[:epoch + 1], 'b', label='Valid')
    plt.axvline(x=best_val_epoch, color='k', linestyle='--', label='best val epoch')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.ylim((1e-4, 0.055))
    plt.legend(fontsize=12)
    plt.title('Losses')

    plt.subplot(2,1,2)
    # plt.plot(epoch_axis, train_seld_scr[:epoch + 1], 'r', label='Train')
    plt.plot(epoch_axis, valid_seld_scr[:epoch + 1], 'b', label='Valid')
    plt.axvline(x=best_val_epoch, color='k', linestyle='--', label='best val epoch')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('SELD scr')
    plt.yscale('log')
    plt.ylim((0.29, 1.003))
    plt.legend(fontsize=12)
    plt.title('SELD score')

    plt.tight_layout()
    fig_name = '{}_{}'.format(log_dir, 'loss_graph.png')
    plt.savefig(fig_name, dpi=150)
    plt.close()

    return train_loss, valid_loss, valid_seld_scr