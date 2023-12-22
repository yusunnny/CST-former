"""
Code from BYsalsa2022
"""
import numpy as np
import torch

from augmentation.data_aug_fnc import ComposeMapTransform, ComposeMapTransform_multi, ComposeTransformNp
from augmentation.data_aug import mixup, mixup_multi, frame_shift,Time_mask, Time_mask_multi, RandomShiftUpDownNp, \
    TfmapRandomSwapChannelFoa, TfmapRandomSwapChannelFoa_multi, \
    TfmapRandomSwapChannelMic, CompositeCutout, FilterAugmentation


from utility.doa_representation import get_accdoa_labels, get_multi_accdoa_doas_4_mixup, \
    inverse_get_multi_accdoa_doas_4_mixup

class data_augment_module:
    def __init__(self, params):
        self.nb_classes = params['unique_classes']
        self.audio_format = params['dataset']
        self.output_format = 'accdoa' if not params['multi_accdoa'] else 'multi_accdoa'

        self.always_apply = params["always_apply_aug"]
        self.is_FS = params["is_FS"]
        self.is_TM = params["is_TM"]
        self.is_FA = params["is_FA"]
        self.is_MU = params["is_MU"]
        if self.is_FA:
            self.db_range = [-3, 3]
            self.filter_type = 'step'
            self.n_band = [3, 6]
            self.min_bw = 6

        self.train_chunk_len = 8 # from BYsalsa2022'
        self.feature_rate = 1 / params['hop_len_s'] # 100
        self.label_rate = 1 / params['label_hop_len_s'] # 10
        self.label_upsample_ratio = int(self.feature_rate / self.label_rate) # 10

        self.train_joint_transform = None
        self.train_transform = None
        self.define_transform()

    def define_transform(self):
        # Mixup transformation (Moderate Mixup)
        if not self.output_format in ['multi_accdoa']:
            self.mixup_transform = ComposeMapTransform([mixup(always_apply=self.always_apply,mixup_label_type='soft'), ])
        else:
            self.mixup_transform_multi = ComposeMapTransform([mixup_multi(always_apply=self.always_apply,mixup_label_type='soft'), ])

        # Data augmentation
        if self.audio_format == 'foa':
            if self.output_format in ['reg_xyz', 'accdoa']:
                if self.is_FS:
                    if self.is_TM:
                        self.train_joint_transform = ComposeMapTransform([
                            TfmapRandomSwapChannelFoa(always_apply=self.always_apply, n_classes=self.nb_classes),
                            frame_shift(always_apply=self.always_apply, label_upsample_ratio=self.label_upsample_ratio),
                            Time_mask(always_apply=self.always_apply, mask_ratios=(10, 20), label_upsample_ratio=self.label_upsample_ratio),
                        ])
                    else:
                        self.train_joint_transform = ComposeMapTransform([
                            TfmapRandomSwapChannelFoa(always_apply=self.always_apply, n_classes=self.nb_classes),
                            frame_shift(always_apply=self.always_apply, label_upsample_ratio=self.label_upsample_ratio),
                        ])
                else:
                    if self.is_TM:
                        self.train_joint_transform = ComposeMapTransform([
                            TfmapRandomSwapChannelFoa(always_apply=self.always_apply, n_classes=self.nb_classes),
                            Time_mask(always_apply=self.always_apply, mask_ratios=(10, 20), label_upsample_ratio=self.label_upsample_ratio),
                        ])
                    else:
                        self.train_joint_transform = ComposeMapTransform([
                            TfmapRandomSwapChannelFoa(n_classes=self.nb_classes),
                        ])
            elif self.output_format in ['multi_accdoa']:
                if self.is_FS:
                    if self.is_TM:
                        self.train_joint_transform = ComposeMapTransform_multi([
                         TfmapRandomSwapChannelFoa_multi(always_apply=self.always_apply, n_classes=self.nb_classes),
                         frame_shift(always_apply=self.always_apply, label_upsample_ratio=self.label_upsample_ratio),
                         Time_mask_multi(always_apply=self.always_apply, mask_ratios=(10, 20), label_upsample_ratio=self.label_upsample_ratio),
                        ])
                    else:
                        self.train_joint_transform = ComposeMapTransform_multi([
                         TfmapRandomSwapChannelFoa_multi(always_apply=self.always_apply, n_classes=self.nb_classes),
                         frame_shift(always_apply=self.always_apply, label_upsample_ratio=self.label_upsample_ratio),
                        ])
                else:
                    if self.is_TM:
                        self.train_joint_transform = ComposeMapTransform_multi([
                         TfmapRandomSwapChannelFoa_multi(always_apply=self.always_apply, n_classes=self.nb_classes),
                         Time_mask_multi(always_apply=self.always_apply, mask_ratios=(10, 20), label_upsample_ratio=self.label_upsample_ratio),
                        ])
                    else:
                        self.train_joint_transform = ComposeMapTransform_multi([
                         TfmapRandomSwapChannelFoa_multi(always_apply=self.always_apply, n_classes=self.nb_classes),
                        ])
            if self.is_FA:
                self.train_transform = ComposeTransformNp([
                    RandomShiftUpDownNp(always_apply=self.always_apply, freq_shift_range=10),  # apply across all channels
                    FilterAugmentation(always_apply=self.always_apply, db_range=self.db_range, n_band=self.n_band, min_bw=self.min_bw, filter_type=self.filter_type,
                                       n_zero_channels=3),
                ])
            else:
                self.train_transform = ComposeTransformNp([
                    RandomShiftUpDownNp(always_apply=self.always_apply, freq_shift_range=10),  # apply across all channels
                ])
        elif self.audio_format == 'mic':
            self.train_joint_transform = ComposeMapTransform([
                TfmapRandomSwapChannelMic(always_apply=self.always_apply, n_classes=self.nb_classes),
            ])
            self.train_transform = ComposeTransformNp([
                RandomShiftUpDownNp(always_apply=self.always_apply, freq_shift_range=10),  # apply across all channels
                CompositeCutout(always_apply=self.always_apply, image_aspect_ratio= self.train_chunk_len / 200,
                                n_zero_channels=3),  # n_zero_channels: these last channels will be replaced with 0
            ])

    def augment_data(self, samples, target):
        # Use inside the dataloader
        X = samples.copy()
        Y = target.copy()
        Y_sed = Y[:, :self.nb_classes]
        Y_doa = Y[:, self.nb_classes:]

        # Apply Augmentation Methods except for the Mixup
        if self.train_joint_transform is not None:
            X, Y_sed, Y_doa= self.train_joint_transform(X, Y_sed, Y_doa)

        if self.train_transform is not None:
            X = self.train_transform(X)

        Y = np.concatenate((Y_sed, Y_doa), axis=-1)

        return X,Y

    def mixup_data(self, samples, target):
        # Use while training
        X = samples
        Y = target
        Y_sed = Y[:, :self.nb_classes]
        Y_doa = Y[:, self.nb_classes:]

        # Mixup can be applied to all array types
        if self.is_MU:
            X, Y_sed, Y_doa = self.mixup_transform(X, Y_sed, Y_doa)

        Y = torch.cat((Y_sed, Y_doa), dim=-2)

        return X,Y

    def aug_and_mix(self, samples, target):

        X = samples
        Y_sed, Y_doa = get_accdoa_labels(target, self.nb_classes)


        # Apply Augmentation Methods except for the Mixup
        if self.train_joint_transform is not None:
            X_new, Y_sed_new, Y_doa_new = [], [], []
            for i in range(X.shape[0]):
                x, y_sed, y_doa = self.train_joint_transform(X[i], Y_sed[i], Y_doa[i])
                X_new.append(x)
                Y_sed_new.append(y_sed)
                Y_doa_new.append(y_doa)
            X = np.asarray(X_new)
            Y_sed = np.asarray(Y_sed_new)
            Y_doa = np.asarray(Y_doa_new)

        if self.train_transform is not None:
            X_new = []
            for i in range(X.shape[0]):
                x = self.train_transform(X[i])
                X_new.append(x)
            X = np.asarray(X_new)

        X, Y_sed, Y_doa = torch.tensor(X), torch.tensor(Y_sed), torch.tensor(Y_doa)

        # Mixup can be applied to all array types
        if self.is_MU:
            X, Y_sed, Y_doa = self.mixup_transform(X, Y_sed, Y_doa)

        return X,Y_doa

    def aug_and_mix_multi(self, samples, target):

        X = samples
        Y_doa = target


        # Apply Augmentation Methods except for the Mixup
        if self.train_joint_transform is not None:
            X_new, Y_doa_new = [], []
            for i in range(X.shape[0]):
                x, y_doa = self.train_joint_transform(X[i], Y_doa[i])
                X_new.append(x)
                Y_doa_new.append(y_doa)
            X = np.asarray(X_new)
            Y_doa = np.asarray(Y_doa_new)

        if self.train_transform is not None:
            X_new = []
            for i in range(X.shape[0]):
                x = self.train_transform(X[i])
                X_new.append(x)
            X = np.asarray(X_new)


        # Mixup can be applied to all array types
        if self.is_MU:
            Y_sed, Y_doa = get_multi_accdoa_doas_4_mixup(Y_doa)
            X, Y_sed, Y_doa = torch.tensor(X), torch.tensor(Y_sed), torch.tensor(Y_doa)
            X, Y_sed, Y_doa = self.mixup_transform_multi(X, Y_sed,Y_doa)
            Y_doa = inverse_get_multi_accdoa_doas_4_mixup(Y_sed,Y_doa)

        else:
            X, Y_doa = torch.tensor(X), torch.tensor(Y_doa)

        return X,Y_doa