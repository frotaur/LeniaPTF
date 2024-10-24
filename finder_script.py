
"""
    Script to run a batched search for transition regions, in between dead and alive phases.
    To use, choose the parameters on top, and potentially modify param_generator, then run the script.
"""
import torch,os, numpy as np
import modules.utils.finder_utils as f_utils
import math, shutil
from time import time
from modules.utils.finder_utils import search_transition
#============================== PARAMETERS ==========================================================

# Where to save the found parameters
folder_save = './data/soliton_search'

device = 'cuda:0'
H,W = 100,100 # Size of the automaton
dt = 0.1 # Time step size
N_steps = 500 # Number of steps to run the automaton for
num_channels = 3 # Number of channels in the automaton (mainly 3 or 1)

num_points = 40 # Number of points to find
refinement = 8 # Number of steps to run the dichotomy search for
cross=False # If True, will compute the transition point between all pairs of parameters. Useful for huge generations, but lessens variations
use_mean = False # If True, uses the mean of the activations to determine death. If False, uses the max.

# threshold below which we say we have found a dead config in the initial search
threshold_e = 0.001
# threshold below which we say we have found a dead config in the dichotomy search (generally matches threshold_e)
threshold_i = 0.001

batch_size = 20 # Number of worlds to simulate in parallel. Reduce if you run out of memory

# Uncomment to use the equivalent of a 'TEMP' directory. IS EMPTIED EACH TIME THE SCRIPT IS RUN
# folder_save= 'data/latest'
batch_folder_save = None # If not None, saves also the batch parameters (generally useless)

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

#=========================== DO NOT MODIFY BELOW THIS LINE ===========================================
if __name__=='__main__':
    from time import time
    # import math

    # if(os.path.exists('data/latest_rand')):
    #     shutil.rmtree('data/latest_rand')
    #     os.makedirs('data/latest_rand',exist_ok=True)


    # if(os.path.exists('data/latest')):
    #     shutil.rmtree('data/latest')
    
    # if(batch_folder_save is not None):
    #     os.makedirs(batch_folder_save, exist_ok=True)

    # os.makedirs(folder_save, exist_ok=True)
    # batch_size = batch_size

    # f_utils.save_rand('data/latest_rand',batch_size=batch_size,num=max(1,20//batch_size),num_channels=num_channels,param_generator=param_generator,device=device)

    with torch.no_grad():
        t00 = time()
        search_transition(save_folder=folder_save,param_generator=param_generator,num_points=num_points,
                          world_size=(H,W), dt=dt, N_steps=N_steps,batch_size=batch_size,
                           thresholds=(threshold_e,threshold_i), refinement=refinement,
                           num_channels=num_channels, device=device)
        # # optimal if sqrt(num_points)>batch_size
        # if(cross):
        #     num_each = math.ceil(math.sqrt(num_points))
        # else:
        #     num_each = num_points

        # for _ in range(math.ceil(num_each/batch_size)):
        #     print(f'Searching for {batch_size} of each phase...')
        #     # find two batches of parameters (one dead one alive)
        #     params_d, params_a = \
        #         f_utils.batch_phase_finder((H,W), dt, N_steps, batch_size=batch_size,params_generator=param_generator, 
        #                                     threshold=threshold_e, num_channels=num_channels,num_examples=min(batch_size,num_each),
        #                                     use_mean=False, device=device) 
            
        #     if(cross):
        #         # Compute transition point between all pairs of parameters
        #         params_d_list = f_utils.param_batch_to_list(params_d,1,squeeze=False)
        #         for param_d in params_d_list:
        #             param_d = f_utils.expand_batch(param_d,params_a['mu'].shape[0])
        #             # Param_d has batch_size = 1, but will broadcast seamlessly when summing with params_a
        #             t_crit, mid_params = f_utils.interest_finder((H,W), dt, N_steps, param_d, params_a, 
        #                                                         refinement, threshold_i, device ,num_channels=num_channels,) 
        #             f_utils.save_param(folder_save, mid_params, batch_folder=batch_folder_save)
        #     else:
        #         t_crit, mid_params = f_utils.interest_finder((H,W), dt, N_steps, params_d, params_a, 
        #                                                         refinement, threshold_i,use_mean=True,device=device,num_channels=num_channels,)

        #         f_utils.save_param(folder_save, mid_params, batch_folder=batch_folder_save)

        print(f'Total time for {num_points} : {time()-t00}')