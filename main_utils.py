import torch
from Automaton import *
import pickle as pk

import numpy as np


def sum_params(params_a, params_d,t_crit, device):
    params = {
        'k_size' : 25,
        'mu' : t_crit*params_a['mu'].to(device) + (1-t_crit)*params_d['mu'].to(device),
        'sigma' : t_crit*params_a['sigma'].to(device) + (1-t_crit)*params_d['sigma'].to(device),
        'beta' : t_crit*params_a['beta'].to(device) + (1-t_crit)*params_d['beta'].to(device),
        'mu_k' : t_crit*params_a['mu_k'].to(device) + (1-t_crit)*params_d['mu_k'].to(device),
        'sigma_k' : t_crit*params_a['sigma_k'].to(device) + (1-t_crit)*params_d['sigma_k'].to(device),
        'weights' : t_crit*params_a['weights'].to(device) + (1-t_crit)*params_d['weights'].to(device)
    }
    return params

## Initialize manually
def gen_params(device):
    """ Generates parameters which are expected to sometime die, sometime live. Very Heuristic."""
    mu = torch.rand((3,3), device=device)
    sigma = mu/(3*np.sqrt(2*np.log(2)))*(1+ (torch.ones_like(mu)-2*torch.rand_like(mu)))
        

    params = {
        'k_size' : 25, 
        'mu':  mu ,
        'sigma' : sigma,
        'beta' : torch.rand((3,3,1), device=device),
        'mu_k' : torch.rand((3,3,1), device=device),
        'sigma_k' : torch.rand((3,3,1), device=device),
        'weights' : torch.rand((3,3), device = device) # element i, j represents contribution from channel i to channel j
    }
    return params

def around_params(params,device):
    """
        Gets parameters which are perturbations around the given set.

        args :
        params : dict of parameters. See LeniaMC for the keys.
    """
    # Rework this
    # Add clamp on dangerous parameters
    # Make variations proportional to current value
    p = {
        'k_size' : 25,
        'mu' : params['mu']*(1 + 0.01*torch.randn((3,3), device=device)),
        'sigma' : torch.clamp(params['sigma']*(1 + 0.01*torch.randn((3,3), device=device)), 0, None),
        'beta' : torch.clamp(params['beta']*(1 + 0.001*torch.randn((3,3,1), device=device)),0,1),
        'mu_k' : params['mu_k']*(1 + 0.001*torch.randn((3,3,1), device=device)),
        'sigma_k' : torch.clamp(params['sigma_k']*(1 + 0.001*torch.randn((3,3,1), device=device)), 0, None),
        'weights' : params['weights']*(1+0.01*torch.randn((3,3), device = device))
    }
    return p

def load_params(file, device):
    """
        Loads and return the parameters given a file (pickle for now) containing them.
    """

    f = open(file, 'rb')
    dico = pk.load(f)
    f.close()

    if len(dico) > 3: # discrimininates between param files before/after LeniaFinder update 
        params = {
            'k_size' : 25,
            'mu' : dico['mu'].to(device),
            'sigma' : dico['sigma'].to(device),
            'beta' : dico['beta'].to(device),
            'mu_k' : dico['mu_k'].to(device),
            'sigma_k' : dico['sigma_k'].to(device),
            'weights' : dico['weights'].to(device)
        }

        exploring = False
        return params, exploring

    else :
        params_d = dico['params_d']
        params_a = dico['params_a']
        t_crit = dico['t_crit']
        exploring = True 
        # Use param sum
        params = sum_params(params_a, params_d, t_crit, device)
        return params, exploring, params_d, params_a, t_crit
    
def compute_ker(auto, device):
    kern = auto.compute_kernel().permute((0,3,2,1))
    maxs = torch.tensor((torch.max(kern[0]), torch.max(kern[1]), torch.max(kern[2])), device=device)
    # print(maxs)
    maxs = maxs[:,None,None,None]
    kern /= maxs 
    return kern