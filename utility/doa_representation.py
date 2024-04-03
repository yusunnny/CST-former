import numpy as np
import torch


########################################################
# Use for Data Augmentation codes
########################################################

def get_multi_accdoa_doas(accdoa_in):
    """
    From single-accdoa format, get SED and DOA output
    Args:
        accdoa_in:  [batch_size, frames, num_track*num_axis*num_class=3*3*13]
        nb_classes: scalar
    Return:
        doaX:       [batch_size, frames, num_axis*nb_overlap*num_class=3*3*13]
    """
    sed = accdoa_in[:,:,0,:]
    x = accdoa_in[:, :, 1,:]
    y = accdoa_in[:, :, 2,:]
    z = accdoa_in[:, :, 3,:]

    doa = np.concatenate((x, y, z), axis=-1)
    return sed, doa

def inverse_get_multi_accdoa_doas(y_sed, y_doa):
    """
    :param y_sed:
    :param y_doa:
    :return: DOA output
    """
    nb_classes = y_doa.shape[-1] // 3
    y_sed = np.expand_dims(y_sed, axis=2)
    y_doa_x = np.expand_dims(y_doa[:, :, 0:nb_classes], axis=2)
    y_doa_y = np.expand_dims(y_doa[:, :, nb_classes:2 * nb_classes], axis=2)
    y_doa_z = np.expand_dims(y_doa[:, :, 2 * nb_classes:], axis=2)
    y_doa = np.concatenate((y_sed, y_doa_x, y_doa_y, y_doa_z), axis=2)

    return y_doa

########################################################
# Use for Data Augmentation Module
########################################################

def get_accdoa_labels(accdoa_in, nb_classes):
    """
    Args:
        accdoa_in:  [batch_size, frames, num_track*num_class=3*13]
        nb_classes: scalar
    Return:
        sed:       [batch_size, frames, num_class=13]
    """
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2 * nb_classes], accdoa_in[:, :, 2 * nb_classes:]
    sed = np.sqrt(x ** 2 + y ** 2 + z ** 2) > 0.5

    return sed, accdoa_in
