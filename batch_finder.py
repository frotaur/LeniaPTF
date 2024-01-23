
"""
    Script to run a batched search for transition regions, in between dead and alive phases.
    To use, choose the parameters on top, and potentially modify param_generator, then run the script.


"""
import torch,os, numpy as np
import modules.utils.b_finder_utils as f_utils
import math, pickle as pk, shutil


folder_save = './data/paper_search'

device = 'cuda:0'
H,W = 200,200
dt = 0.1
N_steps = 600

num_points = 40
refinement = 10
cross=False
# threshold below which we say we have found a dead config in the initial search

# threshold below which we say we have found a dead config in the initial search
threshold_e = 0.03
# threshold below which we say we have found a dead config in the dichotomy search (generally matches threshold_e)
threshold_i = 0.03

use_mean = False
# Uncomment to use latest
folder_save= 'data/latest'

def param_generator(batch_size,device='cpu'):
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
    mu = 0.7*torch.rand((batch_size,3,3), device=device) 
    
    # Std of the grow functions :
    # sigma = mu/(3*np.sqrt(2*np.log(2)))*(1+ (torch.ones_like(mu)-2*torch.rand_like(mu)))
    # sigma = (mu)/(np.sqrt(2*math.log(2)))*(1+torch.clamp(torch.randn((batch_size,3,3), device=device),min=-1+1e-3,max=2))
    sigma = 0.2*torch.rand((batch_size,3,3), device=device)+1e-4
    # sigma = mu/(np.sqrt(2*math.log(2)))*0.7*torch.rand((batch_size,3,3), device=device)+1e-4

    params = {
            'k_size' : 31, 
            'mu':  mu ,
            'sigma' : sigma,
            # Relative sizes of kernel gaussians (l,i,j) represents the l'th ring contribution from channel i to channel j :
            'beta' : torch.rand((batch_size,3,3,3), device=device), 
            # Means of kernel gaussians (3 rings * 3 channels * 3 channels)
            'mu_k' : torch.clamp(0.5+0.3*torch.randn((batch_size,3,3,3), device=device),min=0.,max=1.2), 
            # Stds of kernel gaussians (3 rings * 3 channels * 3 channels)
            'sigma_k' : 0.1*(1+torch.clamp(0.2*torch.randn((batch_size,3,3,3), device=device),min=-1)+1e-4),
            # Weighing of growth functions contribution to each channel
            # 'weights' : torch.rand(batch_size,3,3,device=device)*(1-0.8*torch.diag(torch.ones(3,device=device))) 
            'weights' : torch.rand(batch_size,3,3,device=device)
        }

    return params


# def param_generator(batch_size,device='cpu'):
#     """
#         Prior distribution on the parameters we generate. Can be modified to search in a different
#         space.

#         Args:
#             batch_size : number of parameters to generate
#             device : device on which to generate the parameters
        
#         Returns:
#             dict of batched parameters
#     """
#     # Means of the growth functions :
#     # mu = 0.7*torch.rand((batch_size,3,3), device=device) 
#     mu = 0.7*torch.rand((batch_size,3,3), device=device) 
    
#     # Std of the grow functions :
#     # sigma = mu/(3*np.sqrt(2*np.log(2)))*(1+ (torch.ones_like(mu)-2*torch.rand_like(mu)))
#     # sigma = (mu)/(np.sqrt(2*math.log(2)))*(1+torch.clamp(torch.randn((batch_size,3,3), device=device),min=-1+1e-3,max=2))
#     # sigma = 0.2*torch.rand((batch_size,3,3), device=device)+1e-4
#     sigma = mu/(np.sqrt(2*math.log(2)))*0.7*torch.rand((batch_size,3,3), device=device)+1e-4

#     params = {
#             'k_size' : 31, 
#             'mu':  mu ,
#             'sigma' : sigma,
#             # Relative sizes of kernel gaussians (l,i,j) represents the l'th ring contribution from channel i to channel j :
#             'beta' : torch.rand((batch_size,3,3,3), device=device), 
#             # Means of kernel gaussians (3 rings * 3 channels * 3 channels)
#             'mu_k' : torch.clamp(0.5+0.3*torch.randn((batch_size,3,3,3), device=device),min=0.,max=1.), 
#             # Stds of kernel gaussians (3 rings * 3 channels * 3 channels)
#             'sigma_k' : 0.05*(1+torch.clamp(0.2*torch.randn((batch_size,3,3,3), device=device),min=-1)+1e-4),
#             # Weighing of growth functions contribution to each channel
#             'weights' : torch.rand(batch_size,3,3,device=device)*(1-0.8*torch.diag(torch.ones(3,device=device))) 
#             # 'weights' : torch.rand(batch_size,3,3,device=device)
#         }

#     return params
if __name__=='__main__':
    from time import time
    import math

    batch_folder_save = os.path.join(folder_save,'batch')
    individual_folder_save = os.path.join(folder_save,'individual')

    if(os.path.exists('data/latest_rand')):
        shutil.rmtree('data/latest_rand')
        os.makedirs('data/latest_rand',exist_ok=True)


    if(os.path.exists('data/latest')):
        shutil.rmtree('data/latest')
    
    os.makedirs(batch_folder_save, exist_ok=True)
    os.makedirs(individual_folder_save, exist_ok=True)
    batch_size = 20

    f_utils.save_rand('data/latest_rand',batch_size=batch_size,num=num_points,param_generator=param_generator,device=device)

    with torch.no_grad():
        t00 = time()


        # optimal if sqrt(num_points)>batch_size
        if(cross):
            num_each = math.ceil(math.sqrt(num_points))
        else:
            num_each = num_points

        for _ in range(math.ceil(num_each/batch_size)):
            print(f'Searching for {batch_size} of each phase...')
            # find two batches of parameters (one dead one alive)
            params_d, params_a = \
                f_utils.batch_phase_finder((H,W), dt, N_steps, batch_size=batch_size,params_generator=param_generator, 
                                            threshold=threshold_e, num_examples=min(batch_size,num_each), use_mean=use_mean, device=device) 
            
            if(cross):
                # Compute transition point between all pairs of parameters
                params_d_list = f_utils.param_batch_to_list(params_d,1,squeeze=False)
                for param_d in params_d_list:
                    param_d = f_utils.expand_batch(param_d,params_a['mu'].shape[0])
                    # Param_d has batch_size = 1, but will broadcast seamlessly when summing with params_a
                    t_crit, mid_params = f_utils.interest_finder((H,W), dt, N_steps, param_d, params_a, 
                                                                refinement, threshold_i, device) 
                    f_utils.save_param(batch_folder_save,individual_folder_save, mid_params)
            else:
                t_crit, mid_params = f_utils.interest_finder((H,W), dt, N_steps, params_d, params_a, 
                                                                refinement, threshold_i,use_mean=use_mean,device=device)

                f_utils.save_param(batch_folder_save,individual_folder_save, mid_params)

        print(f'Total time for {num_points} : {time()-t00}')