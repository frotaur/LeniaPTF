
"""
    Script to run a batched search for transition regions, in between dead and alive phases.
    To use, choose the parameters on top, and potentially modify param_generator, then run the script.
"""
import torch, math
import math
from modules.utils.finder_utils import search_transition
from modules.utils import LeniaParams
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

batch_folder_save = None # If not None, saves also the batch parameters (generally useless)

def param_generator(batch_size, num_channels = 3,device='cpu') -> LeniaParams:
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
    sigma = mu/(math.sqrt(2*math.log(2)))*0.8*torch.rand((batch_size,num_channels,num_channels), device=device)+1e-4

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
    
    return LeniaParams(param_dict=params, device=device)

#=========================== DO NOT MODIFY BELOW THIS LINE ===========================================
if __name__=='__main__':
    from time import time


    t00 = time()
    search_transition(save_folder=folder_save,param_generator=param_generator,num_points=num_points,
                        world_size=(H,W), dt=dt, N_steps=N_steps,batch_size=batch_size,
                        thresholds=(threshold_e,threshold_i), refinement=refinement,
                        num_channels=num_channels, device=device)

    print(f'Total time for {num_points} : {time()-t00}')