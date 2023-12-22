import numpy as np
from utility.doa_representation import inverse_get_multi_accdoa_doas, get_multi_accdoa_doas

class ComposeTransformNp:
    """
    Compose a list of data augmentation on numpy array.
    """
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: np.ndarray):
        for transform in self.transforms:
            x = transform(x)
        return x

class DataAugmentNumpyBase:
    """
    Base class for data augmentation for audio spectrogram of numpy array. This class does not alter label
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, x: np.ndarray):
        if self.always_apply:
            return self.apply(x)
        else:
            if np.random.rand() < self.p:
                return self.apply(x)
            else:
                return x

    def apply(self, x: np.ndarray):
        raise NotImplementedError

#############################################################################
# Joint transform
class ComposeMapTransform:
    """
    Compose a list of data augmentation on numpy array. These data augmentation methods change both features and labels.
    """
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: np.ndarray, y_sed: np.ndarray, y_doa: np.ndarray):
        for transform in self.transforms:
            x, y_sed, y_doa = transform(x, y_sed, y_doa)
        return x, y_sed, y_doa

class ComposeMapTransform_multi:
    """
    Compose a list of data augmentation on numpy array. These data augmentation methods change both features and labels.
    """
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: np.ndarray, y_doa: np.ndarray):
        y_sed, y_doa = get_multi_accdoa_doas(y_doa)
        for transform in self.transforms:
            x, y_sed, y_doa = transform(x, y_sed, y_doa)
        y_doa = inverse_get_multi_accdoa_doas(y_sed,y_doa)
        return x, y_doa

class MapDataAugmentBase:
    """
    Base class for joint feature and label augmentation.
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, x: np.ndarray, y_sed: np.ndarray, y_doa: np.ndarray):
        if self.always_apply:
            return self.apply(x=x, y_sed=y_sed, y_doa=y_doa)
        else:
            if np.random.rand() < self.p:
                return self.apply(x=x, y_sed=y_sed, y_doa=y_doa)
            else:
                return x, y_sed, y_doa

    def apply(self, x: np.ndarray, y_sed: np.ndarray, y_doa: np.ndarray):
        """
        :param x: < np.ndarray (n_channels, n_time_steps, n_features)>
        :param y_sed: <np.ndarray (n_time_steps, n_classes)>
        :param y_doa: <np.ndarray (n_time_steps, 3*nclasses)>
        n_channels = 7 for salsa, melspeciv, linspeciv; 10 for melspecgcc, linspecgcc
        """
        raise NotImplementedError
