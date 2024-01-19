import torch,os, pickle as pk, numpy as np
from modules import LeniaMC
import modules.utils.finder_utils as f_utils
from tqdm import tqdm
import math

"""
    Script to find an interesting automaton, and save its parameters. 
    Finds first a dead and an alive automaton, then finds the parameters of the transition in between those.
"""

folder_save = './Data/test_search/'

os.makedirs(folder_save, exist_ok=True)

device = 'cuda:0'
W,H = 250,250
dt = 0.1
N_steps = 300
num_trans = 300
refinement = 5
threshold_e = 0.1 # threshold below which we say we have found a dead config in the initial search
threshold_i = 0.1 # threshold below which we say we have found a dead config in the dychotomy

def param_generator(device):
    """
        Prior distribution on the parameters we generate. Can be modified to search in a different
        space.

    """
    # Means of the growth functions :
    mu = torch.rand((3,3), device=device) 

    # Std of the grow functions :
    # sigma = mu/(3*np.sqrt(2*np.log(2)))*(1+ (torch.ones_like(mu)-2*torch.rand_like(mu)))
    sigma = (mu)/(np.sqrt(2*math.log(2)))*0.95*torch.rand((3,3), device=device)

    params = {
            'k_size' : 25, 
            'mu':  mu ,
            'sigma' : sigma,
            # Relative sizes of kernel gaussians (l,i,j) represents the l'th ring contribution from channel i to channel j :
            'beta' : torch.rand((3,3,3), device=device), 
            # Means of kernel gaussians (3 rings * 3 channels * 3 channels)
            'mu_k' : torch.rand((3,3,3), device=device), 
            # Stds of kernel gaussians (3 rings * 3 channels * 3 channels)
            'sigma_k' : 0.1*(1+0.05*torch.randn((3,3,3), device=device)),
            # Weighing of growth functions contribution to each channel
            'weights' : torch.rand(3,3,device=device)-0.4*torch.diag(torch.rand(3,device=device))  
        }
    
    return params

# Generates num_trans phase transition points
for i in tqdm(range(num_trans),total=num_trans):
    # find two sets of parameters (one dead one alive)
    params_d, params_a = f_utils.phase_finder(W,H, dt, N_steps, param_generator, threshold_e, device) 

    print('param_d before : ', params_d)
    # Find critial point on line connecting them
    t_crit = f_utils.interest_finder(W,H, dt, N_steps, params_d, params_a, refinement, threshold_i, device) 

    print('param_d after : ', params_d)
    dict_out = {'params_d' : params_d, 'params_a' : params_a, 't_crit' : t_crit}

    name = f_utils.hash_dict(dict_out)
    f = open(os.path.join(folder_save,name+'.pk'), "wb") 
    pk.dump(dict_out,f)
    f.close() 
