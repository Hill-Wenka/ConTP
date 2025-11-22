import numpy as np
import pytorch_lightning as L
import torch
from tqdm.notebook import tqdm
from ..config import dict_to_config


class LitModelInference:
    def __init__(self, model_class, data_module_class, ckpt_path=None, device=None, log=True, use_trainer=False):
        self.ckpt_path = ckpt_path
        self.log = log
        self.use_trainer = use_trainer

        if self.log:
            print('[loading checkpoint]:', ckpt_path)

        if isinstance(device, str):
            if device is None or device == 'auto':
                map_location = None
                accelerator = 'auto'
            elif 'cuda' in device:
                # use the default gpu if device == 'cuda'
                map_location = device  # use the specific gpu
                accelerator = 'gpu'
            elif device == 'cpu':
                map_location = 'cpu'
                accelerator = 'cpu'
            else:
                raise ValueError(f'No such pre-defined device {device}')
        elif isinstance(device, torch.device):
            map_location = 'gpu' if device.type == 'cuda' else 'cpu'
            accelerator = 'gpu' if device.type == 'cuda' else 'cpu'
        else:
            map_location = None
            accelerator = 'auto'

        self.ckpt_config = dict_to_config(
            torch.load(self.ckpt_path, map_location=map_location, weights_only=False)['hyper_parameters'])['config']
        self.ckpt_model = model_class.load_from_checkpoint(ckpt_path, map_location=map_location)
        self.device = next(self.ckpt_model.parameters()).device  # auto-detect the used device
        self.map_location = map_location
        self.accelerator = accelerator
        self.ckpt_config.trainer.accelerator = accelerator
        self.ckpt_config.trainer.devices = 10 if str(self.device) == 'cpu' else [int(self.device.index)]
        self.data_module_class = data_module_class
        self.pl_data_module = data_module_class(self.ckpt_config, log=log)

        L.seed_everything(seed=self.ckpt_config.seed, workers=True)
