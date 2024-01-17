import torch
import pygame
from Camera import Camera
from Automaton import *
from finder_utils import *
from hasher import *
import pickle as pk
import numpy as np
from tqdm import tqdm

"""
    Script to find an interesting automaton, and save its parameters. 
    Finds first a dead and an alive automaton, then finds the parameters of the transition in between those.
"""

folder_save = './Data/test_Search/'

os.makedirs(folder_save, exist_ok=True)

device = 'cuda:0'
W,H = 400,400
dt = 0.1
N_steps = 600
num_int = 300
refinement = 5
threshold_e = 1e-6 # threshold below which we say we have found a dead config in the initial search
threshold_i = 0.1 # threshold below which we say we have found a dead config in the dychotomy

def param_range():
    mu = torch.rand((3,3), device=device)
    sigma = mu/(3*np.sqrt(2*np.log(2)))*(1+ (torch.ones_like(mu)-2*torch.rand_like(mu)))
    params = {
            'k_size' : 25, 
            'mu':  mu ,
            'sigma' : sigma,
            'beta' : torch.rand((3,3,3), device=device),
            'mu_k' : torch.rand((3,3,3), device=device),
            'sigma_k' : 0.1*(1+0.05*torch.randn((3,3,3), device=device)),
            'weights' : torch.rand(3,3)-0.4*torch.diag(torch.rand(3))  # element i, j represents contribution from channel i to channel j
        }
    return params

for i in tqdm(range(num_int),total=num_int):
    params_d, params_a = extremum_finder(W,H, dt, N_steps, param_range, threshold_e, device) # find two sets of parameters (one dead one alive)

    t_crit = interest_finder(W,H, dt, N_steps, params_d, params_a, refinement, threshold_i, device) # finds the area of interest on the line params_d -> params_a

    dic_out = {'params_d' : params_d, 'params_a' : params_a, 't_crit' : t_crit}
    # Named through mu and sigma seems reasonable
    # mu = (1-t_crit)*params_d['mu'][0][0].item() + (t_crit)*params_a['mu'][0][0].item()
    # sigma = (1-t_crit)*params_d['sigma'][0][0].item() + (t_crit)*params_a['sigma'][0][0].item()
    name = hash_dict(dic_out)
    f = open(os.path.join(folder_save,name+'.pk'), "wb") 
    pk.dump(dic_out,f)
    f.close() 
