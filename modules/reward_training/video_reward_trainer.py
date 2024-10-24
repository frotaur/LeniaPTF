from .reward_models.vjepa_reward import VJEPAReward
from .reward_trainer import RewardTrainer
from .datasets import DiskRewardDataset

from torch.utils.data import DataLoader, Subset
import torch, torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import os

class VideoRewardTrainer(RewardTrainer):
    """
        Reward trainer for video reward models.
    """

    def __init__(self, model, lr_body=1e-6, no_logging = True, data_loc='video_data', run_name_extra='clipvip', device='cpu'):
        """
            model : nn.Module, the video reward model. Should have a head_params, and a body_params method.
            lr_body : float, learning rate for the body of the model. Pass 0. to freeze the model.
            no_logging : bool, whether to log the training or not
            device : str, device to train the model on
        """        
        model = model.to(device)

        run_config = {'lr_body':lr_body, 'lr_head':1e-3, 'model_config':model.config}

        run_name = f'{run_name_extra}_lr{lr_body:}_head{model.minihead}'
        super().__init__(model=model, data_loc=data_loc, lr_body=lr_body, no_logging=no_logging, 
                         run_config=run_config,device=device, run_name=run_name)

        self.dataset =  DiskRewardDataset(self.data_fold)
        
        self.train_dataset = None # Created on the spot
        self.val_dataset = None # Created on the spot
        
        self.input_shape = model.input_shape

    def create_datapoint(self, data1, data2, annotation) -> str:
        """ 
        Creates a datapoint for the dataset of the image reward model. 

        Args : 
        data1/2 : tensor, (T,3,H,W) representing the video
        annotation : int, 0, 1 or 0.5, representing the annotation of the user

        Returns: 
        str, the path to the saved datapoint
        """

        data1, data2 = self._process_video(data1), self._process_video(data2)

        data_path = f'./{self.data_fold}/{len(os.listdir(self.data_fold))}.pt'
        # Save the data in the appropriate folder
        torch.save({'data1':data1, 'data2':data2, 'annotation':torch.tensor([annotation,1-annotation],dtype=torch.float)}, data_path)
        
        return data_path

    def _process_video(self, tensvid):
        """
            Given a video tensor, returns the processed tensor.

            Args:
            tensvid : (T,3,H,W) representing the video

            Returns:
            (T',3,H',W') tensor, processed tensvid in model's format
        """
        tar_T, _, tar_H, tar_W = self.input_shape
        T = tensvid.shape[0]
        assert tar_T <= tensvid.shape[0], f'tensvid {tensvid.shape[0]} frames, need at least {tar_T} frames'

        # Take tar_T equally spaced frames
        tensvid = tensvid[torch.linspace(0,T-1,tar_T).long()]

        tensvid = torch.einsum('tchw->cthw', tensvid) # interpolate expects channels first
        # Resize the frames
        tensvid = F.interpolate(tensvid, size=(tar_H,tar_W), mode='bilinear')
        tensvid = torch.einsum('cthw->tchw', tensvid) # back to normal
        assert tensvid.shape == self.input_shape, f'tensvid shape {tensvid.shape} not equal to {self.input_shape}'

        return tensvid

    @torch.no_grad()
    def estimate_pair(self, data1, data2, preprocessed=False):
        """
            Estimates the reward for a pair of (un-batched) videos.

            Args:
            data1/2 : (T,3,H,W) tensors representing the videos to compare.	
            preprocessed : bool, whether the data is preprocessed or not

            Returns:
            (2,) tensor of floats, representing the probability of winning for each image.
        """
        if(not preprocessed):
            data1, data2 = self._process_video(data1), self._process_video(data2)

        data1, data2 = data1[None].to(self.device), data2[None].to(self.device)

        probas = torch.stack([self.model(data1),self.model(data2)],dim=1).squeeze() # (2,) 
        probas = F.softmax(probas, dim=0) # (2,), adjusted probabilities

        return probas
    
    def process_batch(self, batch_data):
        data1, data2, annotation = batch_data
        data1 = data1.to(self.device) # (B,T,3,H,W)
        data2 = data2.to(self.device) # (B,T,3,H,W)
        annotation = annotation.to(self.device) # (B,2)

        rewards = torch.stack([self.model(data1),self.model(data2)],dim=1).squeeze(-1) # (B,2) 

        loss = self.unadjusted_cross_entropy(rewards, annotation)
    
        return loss

    def process_batch_valid(self, batch_data):
        data1, data2, annotation = batch_data
        data1 = data1.to(self.device) # (B,T,3,H,W)
        data2 = data2.to(self.device) # (B,T,3,H,W)
        annotation = annotation.to(self.device) # (B,2)

        rewards = torch.stack([self.model(data1),self.model(data2)],dim=1).squeeze(-1) # (B,2) 

        loss = self.unadjusted_cross_entropy(rewards, annotation)
    
        return loss

    def unadjusted_cross_entropy(self, logits, target):
        return F.cross_entropy(logits, target, reduction='mean')

    def get_loaders(self, batch_size, num_workers=0):
        """
            Returns the dataloader for the dataset.
        """
        mixer = torch.randperm(len(self.dataset))

        self.train_dataset = Subset(self.dataset, mixer[:int(0.85*len(mixer))])
        self.val_dataset = Subset(self.dataset, mixer[int(0.85*len(mixer)):])

        train_dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers,shuffle=True)
        valid_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size, num_workers=num_workers,shuffle=True)

        return train_dataloader, valid_dataloader

    def train_model(self, batch_size=10, steps=50):
        """
            Trains the reward model on the dataset created by create_datapoint.
        """
        self.dataset.refresh() # Refreshes the dataset before launching training.
        valid_every = max(1,100//batch_size)

        self.train_steps(steps=steps, batch_size=batch_size, valid_every=valid_every, step_log=2, save_every=1e6, pickup=False,
                         num_workers=4)
        print('Training done !')

