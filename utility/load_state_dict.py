import torch
import numpy as np
from scipy.interpolate import interp2d

def load_state_dict(model, pretrained_model_path):
    print('Initializing the model to the weights - {}'.format(
        pretrained_model_path))
    state_dict = model.state_dict()
    pre_model = torch.load(pretrained_model_path, map_location='cpu')

    if not torch.cuda.device_count()>1:
        best_model_renamed = {}
        for key, value in pre_model.items():
            if key.startswith('module'):
                new_key = key.replace('module.','')
            elif '.module.' in key:
                new_key = key.replace('module.','')
            else:
                new_key = key
            best_model_renamed[new_key] = value
        best_model = {k: v for k, v in best_model_renamed.items() if k in state_dict}
    else:
        best_model = {k: v for k, v in pre_model.items() if k in state_dict}

    state_dict.update(best_model)
    msg = model.load_state_dict(state_dict)
    print(msg)
    return model