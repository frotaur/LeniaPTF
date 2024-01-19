
"""
    Script to run a batched search for transition regions, in between dead and alive phases.
    To use, choose the parameters on top, and potentially modify param_generator, then run the script.


"""
import torch,os, numpy as np
import modules.utils.b_finder_utils as f_utils
import math, pickle as pk


batch_folder_save = './Data/paper_search/batch'
individual_folder_save = './Data/paper_search/individual'

os.makedirs(batch_folder_save, exist_ok=True)
os.makedirs(individual_folder_save, exist_ok=True)

device = 'cuda:0'
H,W = 100,100
dt = 0.1
N_steps = 300

num_points = 50
refinement = 5
 # threshold below which we say we have found a dead config in the initial search
threshold_e = 0.1
# threshold below which we say we have found a dead config in the dichotomy search (generally matches threshold_e)
threshold_i = 0.1 


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
    mu = torch.rand((batch_size,3,3), device=device) 

    # Std of the grow functions :
    # sigma = mu/(3*np.sqrt(2*np.log(2)))*(1+ (torch.ones_like(mu)-2*torch.rand_like(mu)))
    sigma = (mu)/(np.sqrt(2*math.log(2)))*1.05*torch.rand((batch_size,3,3), device=device)

    params = {
            'k_size' : 25, 
            'mu':  mu ,
            'sigma' : sigma,
            # Relative sizes of kernel gaussians (l,i,j) represents the l'th ring contribution from channel i to channel j :
            'beta' : torch.rand((batch_size,3,3,3), device=device), 
            # Means of kernel gaussians (3 rings * 3 channels * 3 channels)
            'mu_k' : 0.5+torch.clamp(torch.randn((batch_size,3,3,3), device=device),min=-0.5,max=0.5), 
            # Stds of kernel gaussians (3 rings * 3 channels * 3 channels)
            'sigma_k' : 0.01*(1+torch.clamp(torch.randn((batch_size,3,3,3), device=device),min=-1)),
            # Weighing of growth functions contribution to each channel
            'weights' : torch.rand(batch_size,3,3,device=device)
        }
    
    return params


if __name__=='__main__':
    from time import time
    import math

    with torch.no_grad():
        t00 = time()
        batch_size = 20
        num_points = 4
        # optimal if sqrt(num_points)>batch_size

        num_each = math.ceil(math.sqrt(num_points))

        for _ in range(math.ceil(num_each/batch_size)):
            print(f'Searching for {batch_size} of each phase...')
            # find two batches of parameters (one dead one alive)
            params_d, params_a = \
                f_utils.batch_phase_finder((H,W), dt, N_steps, batch_size=batch_size,params_generator=param_generator, 
                                            threshold=threshold_e, num_examples=min(batch_size,num_each), device=device) 
            
            # Compute transition point between all pairs of parameters
            # params_a_list = f_utils.param_batch_to_list(params_a,batch_size,squeeze=False)
            params_d_list = f_utils.param_batch_to_list(params_d,1,squeeze=False)
            for param_d in params_d_list:
                param_d = f_utils.expand_batch(param_d,params_a['mu'].shape[0])
                # Param_d has batch_size = 1, but will broadcast seamlessly when summing with params_a
                t_crit, mid_params = f_utils.interest_finder((H,W), dt, N_steps, param_d, params_a, 
                                                            refinement, threshold_i, device) 

                name = f_utils.hash_dict(mid_params)
                if(batch_size>1):
                    f = open(os.path.join(batch_folder_save,name+'.pk'), "wb") 
                    pk.dump(mid_params,f) # Save the resulting parameter batch
                    f.close() 

                mid_params_list = f_utils.param_batch_to_list(mid_params) # Unbatched list of dicts
                for j in range(len(mid_params_list)):
                    f = open(os.path.join(individual_folder_save,name[:-3]+f'{j:02d}'+'.pk'), "wb") 
                    pk.dump(mid_params_list[j],f)
                    f.close()

        print(f'Total time for {num_points} : {time()-t00}')