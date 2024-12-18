import torch
import numpy as np
from showtens import show_image
# def gen_batch_params(batch_size,device='cpu', num_channels=3):
#     """ 
#         Generates reasonably random parameters for Multi-channel lenia.
        
#         Args :
#         batch_size : batch size
#         device : 'cpu' or 'cuda:i', i integer
#         num_channel : number of channels in Lenia
#     """
#     # Growth function parameters :
#     # G_{ij}(x) = 2*e^(-(x-mu_ij)^2/2sigma_ij^2)-1

#     mu = torch.rand((batch_size,num_channels,num_channels), device=device)
#     sigma = mu/(3*np.sqrt(2*np.log(2)))*(1+ (torch.ones_like(mu)-2*torch.rand_like(mu)))
        

#     params = {
#         'k_size' : 25, # size, in pixels, of the Lenia kernel
#         'mu':  mu , # (B,C,C)
#         'sigma' : sigma, # (B,C,C) 
#         'beta' : torch.rand((batch_size,num_channels,num_channels,3), device=device),
#         'mu_k' : torch.rand((batch_size,num_channels,num_channels,3), device=device),
#         'sigma_k' : torch.rand((batch_size,num_channels,num_channels,3), device=device),
#         'weights' : torch.rand((batch_size,num_channels,num_channels), device = device) # element i, j represents contribution from channel i to channel j
#     }

#     return params

# def around_params(params,device):
#     """
#         Gets parameters which are perturbations around the given set.

#         args :
#         params : dict of parameters. See LeniaMC for the keys.
#     """
#     # Rework this
#     # Add clamp on dangerous parameters
#     # Make variations proportional to current value
#     p = {
#         'k_size' : params['k_size'],
#         'mu' : params['mu']*(1 + 0.02*torch.randn((3,3), device=device)),
#         'sigma' : torch.clamp(params['sigma']*(1 + 0.02*torch.randn((3,3), device=device)), 0, None),
#         'beta' : torch.clamp(params['beta']*(1 + 0.02*torch.randn((3,3,1), device=device)),0,1),
#         'mu_k' : params['mu_k']*(1 + 0.02*torch.randn((3,3,1), device=device)),
#         'sigma_k' : torch.clamp(params['sigma_k']*(1 + 0.02*torch.randn((3,3,1), device=device)), 0, None),
#         'weights' : params['weights']*(1+0.02*torch.randn((3,3), device = device))
#     }
#     return p

# def load_params(file, make_batch=False,device='cpu'):
#     """
#         Loads and return the parameters given a file containing them.
#         Silently 'fixes' if the file is unbatched, adds size 1 batch.

#         Args:
#             file : path to the file containing the parameters
#             make_batch : if True, adds a batch dimension to the parameters if not already batched
#             device : device on which to load the parameters
#     """

#     dico = torch.load(file, map_location=device)
#     params = {}

#     mushape = dico['mu'].shape
#     if(len(mushape)==3):
#         make_batch = False

#     # Pure parameter dictionary
#     if('k_size' in dico.keys()):
#         params['k_size'] = dico['k_size']
#         print('loaded k_size : ', params['k_size'])
#     else :
#         params['k_size'] = 31
    
#     for key in dico.keys():
#         if(key!='k_size'):         
#             if(not make_batch):
#                 params[key] = dico[key].to(device)
#             else:
#                 params[key] = dico[key][None,...].to(device)

#     torch.save(params, file) # overwrite with repaired params

#     print(f'LOADED : {file}')
#     return params
        
def compute_ker(auto, device):
    """
        Prepares the kernel and translate it to an RGB image for viewing.
    """
    kern= auto.compute_kernel()[0] # (C,C, k_size, k_size)
    print('Kern shape : ', kern.shape)
    # show_image(kern,rescale=True)
    if(kern.shape[1]==1):
        kern = kern.expand(3,3,-1,-1)
    elif(kern.shape[1]>3):
        kern = kern[:3,:3]
    kern = kern.permute((0,3,2,1)) # (C,k_size,k_size,C)
    maxs = torch.tensor((torch.max(kern[0]), torch.max(kern[1]), torch.max(kern[2])), device=device)
    # print(maxs)
    maxs = maxs[:,None,None,None]
    kern = kern/maxs 
    return kern # (C,k_size,k_size,C)