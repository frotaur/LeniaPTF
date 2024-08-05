import torch
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
def gen_params(device, num_channels=3):
    """ Generates parameters which are expected to sometime die, sometime live. Very Heuristic."""
    mu = torch.rand((num_channels,num_channels), device=device)
    sigma = mu/(3*np.sqrt(2*np.log(2)))*(1+ (torch.ones_like(mu)-2*torch.rand_like(mu)))
        

    params = {
        'k_size' : 25, 
        'mu':  mu ,
        'sigma' : sigma,
        'beta' : torch.rand((num_channels,num_channels,3), device=device),
        'mu_k' : torch.rand((num_channels,num_channels,3), device=device),
        'sigma_k' : torch.rand((num_channels,num_channels,3), device=device),
        'weights' : torch.rand((num_channels,num_channels), device = device) # element i, j represents contribution from channel i to channel j
    }

    return params

def gen_batch_params(batch_size,device='cpu', num_channels=3):
    """ Generates batch parameters."""
    mu = torch.rand((batch_size,num_channels,num_channels), device=device)
    sigma = mu/(3*np.sqrt(2*np.log(2)))*(1+ (torch.ones_like(mu)-2*torch.rand_like(mu)))
        

    params = {
        'k_size' : 25, 
        'mu':  mu ,
        'sigma' : sigma,
        'beta' : torch.rand((batch_size,num_channels,num_channels,3), device=device),
        'mu_k' : torch.rand((batch_size,num_channels,num_channels,3), device=device),
        'sigma_k' : torch.rand((batch_size,num_channels,num_channels,3), device=device),
        'weights' : torch.rand((batch_size,num_channels,num_channels), device = device) # element i, j represents contribution from channel i to channel j
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
        'k_size' : params['k_size'],
        'mu' : params['mu']*(1 + 0.02*torch.randn((3,3), device=device)),
        'sigma' : torch.clamp(params['sigma']*(1 + 0.02*torch.randn((3,3), device=device)), 0, None),
        'beta' : torch.clamp(params['beta']*(1 + 0.02*torch.randn((3,3,1), device=device)),0,1),
        'mu_k' : params['mu_k']*(1 + 0.02*torch.randn((3,3,1), device=device)),
        'sigma_k' : torch.clamp(params['sigma_k']*(1 + 0.02*torch.randn((3,3,1), device=device)), 0, None),
        'weights' : params['weights']*(1+0.02*torch.randn((3,3), device = device))
    }
    return p

def load_params(file, make_batch=False,device='cpu'):
    """
        Loads and return the parameters given a file (pickle for now) containing them.

        Args:
            file : path to the file containing the parameters
            make_batch : if True, adds a batch dimension to the parameters if not already batched
            device : device on which to load the parameters
    """

    f = open(file, 'rb')
    dico = pk.load(f)
    f.close()
    params = {}

    mushape = dico['mu'].shape
    if(len(mushape)==3):
        make_batch = False

    if len(dico) > 3: 
        # Pure parameter dictionary
        if('k_size' in dico.keys()):
            params['k_size'] = dico['k_size']
            print('loaded k_size : ', params['k_size'])
        else :
            params['k_size'] = 31
        
        for key in dico.keys():
            if(key!='k_size'):         
                if(not make_batch):
                    params[key] = dico[key].to(device)
                else:
                    params[key] = dico[key][None,...].to(device)

            

        return params
    else :
        raise ValueError('NOT USING TCRIT IN THE PAPER')
    
def compute_ker(auto, device):
    kern= auto.compute_kernel() # (1,C,C, k_size, k_size)
    if(kern.shape[1]==1):
        kern = kern.expand(-1,3,3,-1,-1)
        return kern[0,:,0,:,:].expand(3,-1,-1)
    elif(kern.shape[1]==2):
        kern = torch.cat((kern,torch.zeros_like(kern[:,1:])),dim=1) # (1,3,2,k_size,k_size)
        kern = torch.cat((kern,torch.zeros_like(kern[:,:,:1])),dim=2) # (1,3,3,k_size,k_size)
    elif(kern.shape[1]>3):
        kern = kern[:,:3,:3]
    kern = (kern.squeeze(0)).permute((0,3,2,1)) # (C,k_size,k_size,C)
    maxs = torch.tensor((torch.max(kern[0]), torch.max(kern[1]), torch.max(kern[2])), device=device)
    # print(maxs)
    maxs = maxs[:,None,None,None]
    kern /= maxs 
    return kern