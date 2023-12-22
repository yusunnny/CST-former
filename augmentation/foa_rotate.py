"""
@ Tho Nguyen, NTU, 2021 04 07
This module includes code to do data augmentation in STFT domain on numpy array:
    1. random volume
    2. random cutout
    3. spec augment
    4. freq shift
==================================================
Example how to use data augmentation
# import
from transforms import CompositeCutout, ComposeTransformNp, RandomShiftUpDownNp, RandomVolumeNp
# call transform
train_transform = ComposeTransformNp([
    RandomShiftUpDownNp(freq_shift_range=10),
    RandomVolumeNp(),
    CompositeCutout(image_aspect_ratio=320 / 128),  # 320: number of frames, 128: n_mels
    ])
# perform data augmentation
X = train_transform(X)  # X size: 1 x n_frames x n_mels
"""
import os
import numpy as np
import torch.nn as nn

class TfmapRandomSwapChannelFoa_positive(nn.Module):
    """
    Positive pair generation for contrastive learning
    This data augmentation random swap xyz channel of tfmap of FOA format.
    """
    def __init__(self, multi_accdoa:bool=False, n_classes: int = 12, m=None):
        super().__init__()
        self.n_classes = n_classes
        self.m = m
        self.multi_accdoa = multi_accdoa

    def apply(self, x: np.ndarray, y_doa: np.ndarray, m):
        """
        :param x < np.ndarray (n_channels, n_time_steps, n_features)>
        :param y_nevent: <np.ndarray (n_time_steps, )>
        Class-wise:
            y_doa: <np.ndarray (n_time_steps, 3*n_classes)> reg_xyz, accdoa
        This data augmentation change x_sed and y_doa
        x feature: x[-3]: Y, x[-2]: Z, x[-1]: X
            W Y Z X Y Z X: 7 channels
        """
        n_input_channels = x.shape[0]
        # assert n_input_channels == 7, 'invalid input channel: {}'.format(n_input_channels)
        x_new = x.copy()
        y_doa_new = y_doa.copy()

        # change input feature
        if m[0] == 1:  # random swap x, y
            x_new[1] = x[3].copy()
            x_new[3] = x[1].copy()
            x_new[-3] = x[-1].copy()
            x_new[-1] = x[-3].copy()
        if m[1] == 1:  # negate x
            x_new[-1] = -x_new[-1].copy()
        if m[2] == 1:  # negate y
            x_new[-3] = -x_new[-3].copy()
        if m[3] == 1:  # negate z
            x_new[-2] = -x_new[-2].copy()

        if not self.multi_accdoa:
            # change y_doa (Single ACCDOA)
            if y_doa.shape[1] == 3 * self.n_classes:  # classwise reg_xyz, accdoa
                if m[0] == 1: # swap x, y
                    y_doa_new[:, 0:self.n_classes] = y_doa[:, self.n_classes:2*self.n_classes].copy()
                    y_doa_new[:, self.n_classes:2*self.n_classes] = y_doa[:, :self.n_classes].copy()
                if m[1] == 1: # negate x
                    y_doa_new[:, 0:self.n_classes] = - y_doa_new[:, 0:self.n_classes].copy()
                if m[2] == 1: # negate y
                    y_doa_new[:, self.n_classes: 2*self.n_classes] = - y_doa_new[:, self.n_classes: 2*self.n_classes].copy()
                if m[3] == 1: # negate z
                    y_doa_new[:, 2*self.n_classes:] = - y_doa_new[:, 2*self.n_classes:].copy()
            else:
                raise NotImplementedError('this output format not yet implemented')
        else:
            # change y_doa (Multi ACCDOA)
            if y_doa.shape[3] == self.n_classes:  # [B, dummy_track, wxyz, classes]
                if m[0] == 1: # swap x, y
                    y_doa_new[:, :, 1, :] = y_doa[:, :, 2,:].copy()
                    y_doa_new[:, :, 2,:] = y_doa[:, :, 1,:].copy()
                if m[1] == 1: # negate x
                    y_doa_new[:, :, 1,:] = - y_doa_new[:, :, 1,:].copy()
                if m[2] == 1: # negate y
                    y_doa_new[:, :, 2,:] = - y_doa_new[:, :,2,:].copy()
                if m[3] == 1: # negate z
                    y_doa_new[:, :, 3,:] = - y_doa_new[:, :, 3,:].copy()
            else:
                raise NotImplementedError('this output format not yet implemented')
        return x_new, y_doa_new, m

#############################################################################

class data_rotate_module:
    def __init__(self, params):
        self.nb_classes = params['unique_classes']
        self.audio_format = params['dataset']
        self.output_format = 'accdoa' if not params['multi_accdoa'] else 'multi_accdoa'
        self.data_rotation = None
        self.define_transform()
        self.foa16rotation = params['FoA16Rotation']
        if self.foa16rotation:
            self.m = [[0,0,1,1], [1,0,1,1], [1,1,0,0], [1,1,0,1], [1,0,0,0], [1,1,1,0], [0,1,0,1],
                      [0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,1],[0,1,1,0], [1,0,1,0],  [0,1,1,1],
                      [1,1,1,1]]
        else:
            self.m = [[0,0,1,1], [1,0,1,1], [1,1,0,0], [1,1,0,1], [1,0,0,0], [1,1,1,0], [0,1,0,1]]

    def define_transform(self):
        # Data augmentation
        if self.audio_format == 'foa':
            if self.output_format in ['reg_xyz', 'accdoa']:
                self.data_rotation = TfmapRandomSwapChannelFoa_positive(n_classes=self.nb_classes)
            elif self.output_format in ['multi_accdoa']:
                self.data_rotation = TfmapRandomSwapChannelFoa_positive(multi_accdoa=True, n_classes=self.nb_classes)

    def save_rotated_data(self, X,Y_doa,feat_dir,label_dir,file_name):
        file_name = file_name.split('.')[0]
        for j in range(len(self.m)):
            feat_save_dir = os.path.join(feat_dir,'{}_ACS{}.npy'.format(file_name,j))
            label_save_dir = os.path.join(label_dir,'{}_ACS{}.npy'.format(file_name,j))
            x, y_doa, _ = self.data_rotation.apply(X, Y_doa, self.m[j])
            x = x.transpose(1,0,2)
            x = np.reshape(x,(x.shape[0], -1))
            np.save(feat_save_dir,x)
            np.save(label_save_dir,y_doa)


    def rotate_data(self, X, Y_doa):
        # Apply FoA Rotation
        if self.rotate_data is not None:
            X_new, Y_doa_new, rot_idx_all = [], [], []
            for i in range(X.shape[0]):
                for j in range(len(self.m)):
                    x, y_doa, rot_idx = self.data_rotation.apply(X[i], Y_doa[i], self.m[j])
                    X_new.append(x)
                    Y_doa_new.append(y_doa)
                    rot_idx_all.append(rot_idx)
            X = np.asarray(X_new)
            Y_doa = np.asarray(Y_doa_new)
            rot_idx = np.asarray(rot_idx_all)

        return X,Y_doa,rot_idx