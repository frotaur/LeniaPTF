from .vit import vit_large,vit_tiny
import torch


def get_vit_large(save_weights_file=None, num_frames=16):
    model = vit_large(num_frames=num_frames)

    if save_weights_file is not None:
        state_dict = _vit_state_dict(save_weights_file)
        model.load_state_dict(state_dict)
        print('Load successful, nice!')
    return model

def get_vit_tiny(save_weights_file=None, num_frames=16):
    model = vit_tiny(num_frames=num_frames)

    if save_weights_file is not None:
        model.load_state_dict(torch.load(save_weights_file))
        print('Load successful, nice!')
    
    return model

def _vit_state_dict(file_path):
    original_state_dict = torch.load(file_path)['encoder']
    new_dict = {}
    for key, value in original_state_dict.items():
        # Remove the 'module.backbone.' prefix
        new_key = key.replace('module.backbone.', '')
        # Add the modified key and its corresponding value to the new dictionary
        new_dict[new_key] = value

    return new_dict