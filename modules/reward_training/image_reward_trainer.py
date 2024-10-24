from .reward_models.squeezenet import SqueezeReward
from .reward_trainer import RewardTrainer
from .datasets import MemoryRewardDataset

from torch.utils.data import DataLoader, Subset
import torch, torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import os, random
from torchvision import transforms
import wandb


class ImageRewardTrainer(RewardTrainer):
    """
        Reward trainer based on a SqueezeNet model, which
        is used to create a reward model for images.
    """

    def __init__(self, lr_body=0., no_logging = True, device='cpu'):
        model = SqueezeReward(device)
        run_name = f'image_lr{lr_body}'

        super().__init__(model=model, data_loc='image_data', lr_body=lr_body,
                          no_logging=no_logging,run_name=run_name, device=device)

        self.dataset =  MemoryRewardDataset(self.data_fold)
        
        self.train_dataset = None # Created on the spot
        self.val_dataset = None # Created on the spot

    def create_datapoint(self, data1, data2, annotation) -> str:
        """ 
        Creates a datapoint for the dataset of the image reward model. 

        Args : 
        data1/2 : tensor, (T,3,H,W) or (3,H,W) tensor representing the video or last frame
        annotation : int, 0, 1 or 0.5, representing the annotation of the user

        Returns: 
        str, the path to the saved datapoint
        """

        data1, data2 = self._process_video(data1), self._process_video(data2)

        data_path = f'./{self.data_fold}/{len(os.listdir(self.data_fold))}.pt'
        # Save the data in the appropriate folder
        torch.save({'data1':data1, 'data2':data2, 'annotation':torch.tensor([annotation,1-annotation],dtype=torch.float)}, data_path)
        
        return data_path
    
    def _process_video(self, data):
        """
            Returns the image from the data, which can be a video or a single frame.

            Args:
            data : (T,3,H,W) or (3,H,W) tensor representing the video or last frame

            Returns:
            (3,H,W) tensor, representing the image.
        """
        # Use clone() to avoid memory issues when saving
        if len(data.shape) == 4:
            return data[-1].clone()
        return data

    @torch.no_grad()
    def estimate_pair(self, data1, data2):
        """
            Estimates the reward for a pair of (batched) images.

            Args:
            data : (3,H,W) or (T,3,H,W) tensors representing the videos or last frame to compare.

            Returns:
            (2,) tensor of floats, representing the probability of winning for each image.
        """

        img1, img2 = self._process_video(data1)[None].to(self.device), self._process_video(data2)[None].to(self.device)
        rewards = torch.stack([self.model(img1),self.model(img2)],dim=1).squeeze() # (2,) 
        rewards = F.softmax(rewards, dim=0) # (2,), adjusted probabilities

        return rewards
    
    def process_batch(self, batch_data):
        data1, data2, annotation = batch_data
        data1 = data1.to(self.device) # (B,3,H,W)
        data2 = data2.to(self.device) # (B,3,H,W)
        annotation = annotation.to(self.device) # (B,2)

        rewards = torch.stack([self.model(data1),self.model(data2)],dim=1).squeeze(-1) # (B,2) 

        loss = self.unadjusted_cross_entropy(rewards, annotation)
        
        self.logger.log({'lr':self.scheduler.get_last_lr()[0]},commit=False)
        return loss

    def process_batch_valid(self, batch_data):
        data1, data2, annotation = batch_data
        data1 = data1.to(self.device) # (B,3,H,W)
        data2 = data2.to(self.device) # (B,3,H,W)
        annotation = annotation.to(self.device) # (B,2)

        rewards = torch.stack([self.model(data1),self.model(data2)],dim=1).squeeze(-1) # (B,2) 

        loss = self.unadjusted_cross_entropy(rewards, annotation)
    
        return loss

    def valid_log(self):
        data1, data2, annotation = self.val_dataset[random.randint(0,len(self.val_dataset)-1)]
        # Convert tensors to images
        transform = transforms.ToPILImage()
        image1 = transform(data1)
        image2 = transform(data2)
        estimate = self.estimate_pair(data1, data2)
        # Log images with annotations
        self.logger.log({
            "Validation/Image1": wandb.Image(image1, caption=f'{estimate[0]}%, vs ann : {annotation}'),
            "Validation/Image2": wandb.Image(image2, caption=f'{estimate[1]}%, vs ann : {1-annotation}')
        })



    def adjusted_cross_entropy(self, logits, target):
        probas = F.softmax(logits, dim=1)
        probas = 0.9*probas + 0.05 # insert a 'randomness' of 10%
        
        return F.nll_loss(probas.log(), target, reduction='mean') # Doesn't work for soft labels

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

    def train_model(self, steps=50, batch_size=10):
        """
            Trains the reward model on the dataset created by create_datapoint.
        """
        self.dataset.refresh() # Refreshes the dataset before launching training.
        valid_every = max(1,200//batch_size)
        self.train_steps(steps=steps, batch_size=batch_size, valid_every=valid_every, step_log=2, save_every=1e6, pickup=False)
        print('Training done !')