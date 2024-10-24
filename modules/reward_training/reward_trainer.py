from torchenhanced import ConfigModule, Trainer
import torch, torch.nn as nn, torchvision
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import json, os
from tqdm import tqdm


class RewardTrainer(Trainer):
    """
        Base class, wrapper for a reward model trainer.
        Contains utility methods to train a reward model, 
        given annotation on a dataset.

        Idea :
        create_datapoint :
        method which will be called
        each time an annotation is made. The argument will be basically
        the two datas that are shown (video, image, whatever), and the
        annotation for which is preferred. This function should take that information, 
        and create a datapoint in the appropriate folder. It will then use this folder
        to create a dataset, which will be used to train a model.
    
        train_model :
        this simply makes a training run of the reward model. We'll see if I can use
        the native 'train_steps' from torchenhanced, if not, I'll just re-implement it
        completely. I'll still inherit from trainer I think, to have access to the saving
        and loading of the model, which might be useful.

        estimate_pair :
        given a pair of datas as in 'create_datapoint', this should return the estimation
        from the model. Depending on the model, this might differ. It will be used by the 
        main program to sort the images to show, to show only ones which are unsure (probably, to be seen).


    """

    def __init__(self, model : nn.Module, data_loc, optimizer=None, scheduler=None, lr_body=0., no_logging=True, run_config={}, run_name=None, device='cpu'):
        """
            Args:
            model : model to be trained. Probably to be hardcoded in inheriting classes
            data_loc : str, location where the dataset data will be saved
            optimizer : optimizer to use. If None, will use AdamW with lr=1e-3 for the head, and lr=lr_body for the body.
            scheduler : scheduler to use. If None, will use a LinearLR with start_factor=1e-5, end_factor=1, total_iters=300.
            lr_body : float, learning rate for the body of the model. If 0., will freeze the body.
            no_logging : bool, whether to log the training or not
            run_config : dict, configuration for the run. Will be saved in the wandb run.
            device : str, device to train the model on
        """

        body_params = model.body_params()

        if(optimizer is None):
            if(lr_body<=1e-8):
                for param in body_params:
                    param.requires_grad = False
                optim = AdamW(model.head_params(),lr=1e-3)
                lr_body= 0.
            else :
                optim = AdamW([
                    {'params': body_params , 'lr': lr_body},
                    {'params': model.head_params(), 'lr': 1e-3}
                ])

        if(scheduler is None):
            schedu = LinearLR(optim,start_factor=1e-5, end_factor=1, total_iters=300)

        run_config = {'lr_body':lr_body, 'lr_head':1e-3, 'model_config':model.config}
        super().__init__(model=model, optim=optimizer, scheduler=scheduler, save_loc='lenia_rlhf',project_name='reward_train',device=device,
                         no_logging=no_logging, run_config=run_config, run_name=run_name)
        
        self.data_fold = data_loc

        os.makedirs(data_loc, exist_ok=True)

    def create_datapoint(self, data1, data2, annotation) -> str:
        """
            Create a datapoint from the two datas given, provided the annotation.
            Returns string for the location of the datapoint.
            TO BE REDEFINED IN INHERITING CLASSES.
        """
        raise NotImplementedError('Method create_datapoint must be implemented in inheriting class.')

    def train_model(self):
        """
            Train the model on the dataset created by create_datapoint.
            TO BE REDEFINED IN INHERITING CLASSES.
        """
        raise NotImplementedError('Method train_model must be implemented in inheriting class.')
    
    def estimate_pair(self, data):
        """
            Estimate the reward for the data given.
            TO BE REDEFINED IN INHERITING CLASSES.
        """
        raise NotImplementedError('Method estimate_pair must be implemented in inheriting class.')
    
    def _process_video(self, vidtens):
        """
            Given a video tensor, returns the processed tensor.
        """
        return vidtens
    
    def annotation_to_data(self, annotations_file:str, video_folder:str):
        """
            Given an annotations file, and a data folder, will create the datapoints corresponding to the annotations.
            NOTE : right now, this works only if the video_folder contains mp4 videos. To be chnaged in the future, if needed.
        """
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        print('Creating dataset...')
        missing =0 
        for annotation in tqdm(annotations):
            dataname1 = f"{video_folder}/{annotation['left']}.mp4"
            dataname2 = f"{video_folder}/{annotation['right']}.mp4"
            score = annotation['side']
            
            try:
                data1 = (torchvision.io.read_video(dataname1, output_format='TCHW', pts_unit='sec')[0]).float() / 255.
                data2 = (torchvision.io.read_video(dataname2, output_format='TCHW', pts_unit='sec')[0]).float() / 255.
            except FileNotFoundError:
                print(f'File not found: {dataname1}.mp4 or {dataname2}.mp4. Skipping this pair.')
                print(f'REMOVING pair from annotation, as it is outdated')
                annotations.remove(annotation)
                missing+=1
                continue

            self.create_datapoint(data1, data2, score)
        
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f)
        
        print(f'Dataset created! Removed {missing} missing pairs from the annotations file.')
    
    def video_to_tensor(self, video_path:str):
        """
            Given a video path, will return the video as a tensor.
        """
        video = (torchvision.io.read_video(video_path, output_format='TCHW', pts_unit='sec')[0]).float() / 255.
        
        return self._process_video(video)
    @property
    def num_datapoints(self):
        """
            Returns the number of datapoints in the dataset.
        """
        return len(os.listdir(self.data_fold))