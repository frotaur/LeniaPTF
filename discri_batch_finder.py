
"""
    Script to run a batched search for transition regions, in between dead and alive phases.
    To use, choose the parameters on top, and potentially modify param_generator, then run the script.
"""
import torch,os, numpy as np
import modules.utils.b_finder_utils as f_utils
import math, pickle as pk, shutil


#============================== PARAMETERS ==========================================================

# Where to save the found parameters
folder_save = './data/discribois'

device = 'cuda:0'
H,W = 100,100 # Size of the automaton

N_steps = 800 # Number of steps to run the automaton for

DISCRI = 13 # Discretization of the automaton
num_points = 75 # Number of points to find
refinement = 8 # Number of steps to run the dichotomy search for
cross=False # If True, will compute the transition point between all pairs of parameters. Useful for huge generations, but lessens variations
use_mean = True # If True, uses the mean of the activations to determine death. If False, uses the max.

# threshold below which we say we have found a dead config in the initial search
threshold_e = 0.2
# threshold below which we say we have found a dead config in the dichotomy search (generally matches threshold_e)
threshold_i = 0.2

batch_size = 20 # Number of worlds to simulate in parallel. Reduce if you run out of memory

# Uncomment to use the equivalent of a 'TEMP' directory. IS EMPTIED EACH TIME THE SCRIPT IS RUN
folder_save= 'data/latest'


def param_generator(batch_size, device='cpu'):
    mu = 0.7*torch.rand((batch_size,1), device=device)
    sigma = 0.15*torch.rand_like(mu)+1e-5
            

    params = {
        'k_size' : 27, 
        'mu':  mu ,
        'sigma' : sigma,
        'mu_k' : torch.full((batch_size,), fill_value=0.5, device=device),
        'sigma_k' : torch.full((batch_size,),fill_value=0.15, device=device),
    }
    
    return params

#=========================== DO NOT MODIFY BELOW THIS LINE ===========================================
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
    batch_size = batch_size

    f_utils.save_discri_rand('data/latest_rand',batch_size=batch_size,num=max(1,20//batch_size),param_generator=param_generator,device=device)

    crit_point_list = []
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
                f_utils.discri_batch_phase_finder((H,W),  N_steps,DISCRI, batch_size=batch_size,params_generator=param_generator, 
                                            threshold=threshold_e, num_examples=min(batch_size,num_each), use_mean=use_mean, device=device) 
            
            if(cross):
                # Compute transition point between all pairs of parameters
                params_d_list = f_utils.param_batch_to_list(params_d,1,squeeze=False)
                for param_d in params_d_list:
                    param_d = f_utils.expand_batch(param_d,params_a['mu'].shape[0])
                    # Param_d has batch_size = 1, but will broadcast seamlessly when summing with params_a
                    t_crit, mid_params = f_utils.discri_interest_finder((H,W), DISCRI, N_steps, param_d, params_a, 
                                                                refinement, threshold_i, device ) 
                    f_utils.save_param(batch_folder_save,individual_folder_save, mid_params)
            else:
                t_crit, mid_params = f_utils.discri_interest_finder((H,W), DISCRI, N_steps, params_d, params_a, 
                                                                refinement, threshold_i,use_mean=use_mean,device=device)

                f_utils.save_param(batch_folder_save,individual_folder_save, mid_params)
            crit_mu = mid_params['mu']
            crit_sigma = mid_params['sigma']
            crit_point_list.extend([(mu,sig) for mu,sig in zip(crit_mu.tolist(),crit_sigma.tolist())])

        print(f'Total time for {num_points} : {time()-t00}')

        with open(os.path.join('.','crit_points_list.pk'),'wb') as f:
            pk.dump(crit_point_list,f)