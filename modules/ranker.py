from .utils.finder_utils import search_transition
from .utils.hash_params import params_to_words
from .reward_training import VideoRewardTrainer
from .Automaton import BatchLeniaMC
import torch, os, shutil
import json
from tqdm import tqdm
from showtens import show_image, save_video


class Ranker():
    """
        Helper class to generate interesting parameters, combining a
        trained reward model and PTF search.
    """

    def __init__(self, reward_trainer:VideoRewardTrainer, evolve_steps=1000, device='cpu'):
        """
            Args:
                reward_trainer : pretrained Reward Trainer,
                param_generator : function, generates parameters
                device : str, device 
        """
        self.reward_trainer = reward_trainer
        self.param_generator = param_generator
        self.device = device

        self.reward_model = reward_trainer.model
        self.evolve_steps = evolve_steps

        # HACKY but to avoid super memory when dealing with the videos
        tar_frames = self.reward_model.input_shape[0]

        self.allowed_frames = set([i.item() for i in torch.linspace(0,self.evolve_steps,tar_frames).long()])

    @torch.no_grad()
    def generate_interesting_params(self, output_folder, score_cutoff, param_generator, 
                                    search_params=None,use_ptf=True, save_bad=False):
        """
            Generates interesting parameters, by ranking them with the reward model.
            Optionally screens the parameters with PTF.

            Args:
            output_folder : str, where to save the parameters
            score_cutoff : float, minimum score to keep the parameters
            param_generator : function, generates parameters
            search_params : str, path to the search configuration
            use_ptf : bool, whether to use PTF or not
            save_bad : bool, whether to save the parameters that did not pass the cutoff
        """
        self.reward_model.eval()
        if(os.path.exists(output_folder)):
            print('Warning, output folder already exists, will add new data to it')
        
        tempfolder = 'ranker_temp'
        if(not os.path.exists(search_params)):
            raise ValueError('Search parameters file not found')

        with open(search_params,'r') as f:
            search_params = json.load(f)

        batch_size = search_params['batch_size']
        
        garbage_folder = os.path.join(tempfolder,'params')
        params_folder = os.path.join(tempfolder,'params_batch')

        os.makedirs(params_folder,exist_ok=True)
        os.makedirs(garbage_folder,exist_ok=True)

        print('Generating ranking candidates...')
        # Generates the candidates :
        if(use_ptf):
            search_transition(save_folder=garbage_folder,save_batch_params=True,param_generator=param_generator,**search_params)
        else:
            num_points = search_params['num_points']
            for i in range(num_points//batch_size):
                params = param_generator(batch_size,
                                         search_params['num_channels'],
                                         device=search_params['device'])
                params.save(folder=garbage_folder)
                save_param(folder=garbage_folder,params=params, batch_folder=params_folder)
        
        print('Ranking candidates...')
        simulator = self.get_simulator(search_params)

        batch_param_files = [os.path.join(params_folder,file) for file in os.listdir(params_folder)]

        os.makedirs(output_folder,exist_ok=True)
        if(save_bad):
            bad_folder = output_folder+'_bad'
            os.makedirs(bad_folder,exist_ok=True)
        
        for _,file in tqdm(enumerate(batch_param_files),total=len(batch_param_files)):
            batch_params = torch.load(file)
            scores = self.score_params(batch_params, simulator, repetitions=1) # (B,)
            passed = (scores>score_cutoff) # (B,) mask
            out_params = self._filter_params(batch_params, passed)

            scores = [ f'{scor.item():.2f}' for scor in scores[passed]]
            if(len(out_params)>0):
                save_param(folder=output_folder, params=out_params, annotation=scores)

            if(save_bad):
                out_params = self._filter_params(batch_params, ~passed)
                scores = [ f'{scor.item():.2f}' for scor in scores[~passed]]
                save_param(folder=bad_folder, params=out_params)

        # Delete all the temporary files
        shutil.rmtree(tempfolder)
        print('Ranking finished, parameters saved in output folder')
    

    def get_simulator(self, search_params):
        batch_size = search_params['batch_size']
        rank_batch_size = search_params.get('rank_batch_size',batch_size)
        B,H,W = (rank_batch_size, *search_params['rank_world_size'])
        # Generates the video tensors :
        simulator = BatchLeniaMC(size = (B,H,W) , num_channels=search_params['num_channels'],
                                 device=search_params['device'], dt=search_params['dt'], use_fft=True)
        return simulator

    def score_params(self, batch_params, simulator:BatchLeniaMC, repetitions=1):
        """
            Scores the batch of parameters, averaging over repetitions.

            Args:
                batch_params : dict of batched parameters
                simulator : BatchLeniaMC, simulator
                repetitions : int, number of repetitions of scoring (done sequentially)
        """
        batch_size = batch_params['mu'].shape[0]

        scores = []
        for _ in range(repetitions):
            tensors = self._params_to_tensor(batch_params,simulator) # (B,T,C,H,W) ready for reward model
            scores.append(self.reward_model(tensors)) # (B,)
        scores = torch.stack(scores,dim=1).mean(dim=1) # (B,)

        return scores

    def _params_to_tensor(self, parameters, simulator:BatchLeniaMC):
        """
            Given a Lenia parameter state_dict, returns a (B,T,C,H,W)
            tensor in the correct format for the reward model, by
            evolving the automaton.

            Args:
            parameters : lenia state_dict
            simulator : Automaton class used to evolve the parameters
        """
        simulator.update_params(parameters)
        simulator.set_init_fractal()
        states = []


        for i in range(self.evolve_steps+1):
            if(i in self.allowed_frames):
                states.append(simulator.state) # (B,C,H,W)
            simulator.step()
    
        states = torch.stack(states,dim=1) # (B,T,C,H,W)
        states = self.reward_trainer._process_video(states) # (B,T',C,H',W')

        return states

    def _filter_params(self, batch_params, passed_mask):
        """
            Filters the parameters that passed the score cutoff.

            Args:
            batch_params : dict of batched parameters
            passed_mask : (B,) bool, which parameters passed

            Returns:
            list of dicts, parameters that passed
        """
        out_params = {}
        for key in batch_params.keys():
            if (key=='k_size'):
                out_params[key] = batch_params[key]
            else:
                out_params[key] = batch_params[key][passed_mask] # shape (B',...)
        
        return out_params



def param_generator(batch_size, num_channels = 3,device='cpu'):
    """
        Prior distribution on the parameters we generate. Can be modified to search in a different
        space.

        Args:
            batch_size : number of parameters to generate
            device : device on which to generate the parameters
        
        Returns:
            dict of batched parameters
    """
    # Means of the growth functions :
    # mu = 0.7*torch.rand((batch_size,3,3), device=device) 
    mu = 0.7*torch.rand((batch_size,num_channels,num_channels), device=device) 
    
    # Std of the grow functions :
    # sigma = mu/(3*np.sqrt(2*np.log(2)))*(1+ (torch.ones_like(mu)-2*torch.rand_like(mu)))
    # sigma = (mu)/(np.sqrt(2*math.log(2)))*(1+torch.clamp(torch.randn((batch_size,num_channels,num_channels), device=device),min=-1+1e-3,max=2))
    # sigma = 0.2*torch.rand((batch_size,num_channels,num_channels), device=device)+1e-4
    sigma = mu/(np.sqrt(2*math.log(2)))*0.8*torch.rand((batch_size,num_channels,num_channels), device=device)+1e-4

    params = {
            'k_size' : 31, 
            'mu':  mu ,
            'sigma' : sigma,
            # Relative sizes of kernel gaussians (l,i,j) represents the l'th ring contribution from channel i to channel j :
            'beta' : torch.rand((batch_size,num_channels,num_channels,3), device=device), 
            # Means of kernel gaussians (3 rings * 3 channels * 3 channels)
            # 'mu_k' : torch.clamp(0.5+0.3*torch.randn((batch_size,num_channels,num_channels,3), device=device),min=0.,max=1.), 
            'mu_k' : torch.clamp(0.5+0.2*torch.randn((batch_size,num_channels,num_channels,3), device=device),min=0.,max=1.2), 
            # Stds of kernel gaussians (3 rings * 3 channels * 3 channels)
            'sigma_k' : 0.05*(1+torch.clamp(0.3*torch.randn((batch_size,num_channels,num_channels,3), device=device),min=-0.9)+1e-4),
            # Weighing of growth functions contribution to each channel
            'weights' : torch.rand(batch_size,num_channels,num_channels,device=device)*(1-0.8*torch.diag(torch.ones(num_channels,device=device)))
            # 'weights' : torch.rand(batch_size,3,3,device=device)
        }
    
    return params
