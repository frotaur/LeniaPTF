import torch, hashlib
from ..Automaton import BatchLeniaMC
from time import time
import copy, os, pickle as pk
from tqdm import tqdm
from math import ceil

@torch.no_grad()
def batch_phase_finder(size, dt, N_steps, batch_size, params_generator, threshold,num_examples=None,  device='cpu'):
    """
        Finds a set of parameter of dead automaton, and a set of parameters of an alive automaton.

        Args:
            size : (H,W) height and width of world
            dt : time step
            N_steps : number of simulation steps before checking phase
            params_generator : function which returns a batch of parameters
            threshold : threshold below which we say we have found a dead config
            num_examples: Number of params of each phase to find.
                If None, generates batch_size//2 dead and batch_size//2 alive automata.
            device : device on which to run the automaton
        
        Returns: 2-uple (dead_params,alive_params)
            dead_params : Dictionary of pameters, batch_size is num_examples
            alive_params : Dictionary of pameters, batch_size is num_examples
    """

    found_d = False
    found_a = False
    H,W = size

    auto = BatchLeniaMC((batch_size,*size), dt, device=device)
    auto.to(device)

    if(num_examples is None):
        num_examples = int(batch_size//2)

    dead_params = {}
    alive_params = {}
    params = params_generator(batch_size,device)

    dead_params['k_size'] = params['k_size']
    alive_params['k_size'] = params['k_size']

    n_dead = 0
    n_alive = 0
    while (not(found_d and found_a)):
        # Initialize the automaton
        params = params_generator(batch_size,device) # Params for next run TODO : ACTUALLY PUT IT UP THERE
        auto.update_params(params)
        auto.set_init_perlin()

        t0 = time()
        for _ in range(N_steps):
            auto.step()
        print('Simulation took : ', time()-t0)	
        mass_f = auto.mass().max(dim=1).values #  (B,3)
        dead_mask = mass_f < threshold # (B,) True if dead
        num_d = dead_mask.sum().item() # Number of dead examples in batch
        num_a = (~dead_mask).sum().item()
        print('Dead masses : ', mass_f[dead_mask])
        print('Alive masses : ', mass_f[~dead_mask])
        print(f'Found {num_d} dead and {num_a} alive')
        dead_add = min(num_examples-n_dead,num_d) # Number of dead examples to keep to reach num_examples
        alive_add = min(num_examples-n_alive,num_a) # Number of alive examples to keep to reach num_examples


        for key,cur_param in params.items():
            if(key!='k_size'):
                B = cur_param.shape[0]
                par_size = cur_param.shape[1:]

                param_d = cur_param[dead_mask] # (Flattened selection)
                param_a = cur_param[~dead_mask] # (Flattened selection)

                param_d = param_d.reshape(-1,*par_size)[:dead_add] # (num_D,par_size)
                param_a = param_a.reshape(-1,*par_size)[:alive_add] # (num_A,par_size)

                if(key in dead_params):
                    dead_params[key] = torch.cat((dead_params[key],param_d),dim=0) # (n_dead+num_D,par_size)
                else:
                    dead_params[key] = param_d

                if(key in alive_params):
                    alive_params[key] = torch.cat((alive_params[key],param_a),dim=0) # (n_alive+num_A,par_size)
                else:
                    alive_params[key] = param_a

        print(f'Adding {dead_add} dead')
        n_dead += dead_add # Num of dead configurations found
        print(f'Adding {alive_add} alive')
        n_alive += alive_add # Num of alive configurations found

        if (n_dead >= num_examples):
            print(f'Found all {n_dead} dead')
            found_d = True
        else:
            print('Continuing search for dead, remain ', num_examples-n_dead, ' to find')
        if (n_alive >= num_examples):
            print(f'Found all {n_alive} alive')
            found_a=True
        else:
            print('Continuing search for alive, remain ', num_examples-n_alive, ' to find')
    
    return dead_params, alive_params
       

@torch.no_grad()
def interest_finder(size, dt, N_steps, p_dead, p_alive, refinement, threshold, device='cpu'):
    """
        By dichotomy, finds the parameters of an interesting automaton. By interesting, here
        we mean a set of parameters which lies at the transition between an asymptotically dead
        and an asymptotically alive automaton. NOTE ! Will simulate all of the provided p_dead and p_alive,
        so make sure they fit in memory.

        Args :
            size : (H,W) height and width of world
            dt : time step
            N_steps : number of steps the automaton goes through for each set of parameters
            device : device on which to run the automaton
            p_dead : batch of parameters of a dead automaton. Batch_size much match params_a  (dict)
            p_alive : batch of parameters of an alive automaton. Batch_size much match params_d  (dict)
            refinement : number of iterations of dichotomy
            threshold : threshold below which we say we have a dead config
        
        Returns:
            t_crit : threshold for which we have a transition between dead and alive
            mid_params : parameters of the automaton at the transition
    """
    print('Computing dichotomy on found phases')
    p_d = copy.deepcopy(p_dead)
    p_a = copy.deepcopy(p_alive)

    batch_size = p_a['mu'].shape[0]
    assert batch_size==p_d['mu'].shape[0], 'Batch sizes must match'

    t_crit = torch.full((batch_size,),0.5,device=device)
    auto = BatchLeniaMC((batch_size,*size), dt ,device=device)
    auto.to(device)

    for i in tqdm(range(refinement)):
        mid_params = mean_params(p_d,p_a)

        auto.update_params(mid_params)
        auto.set_init_perlin()

        print('Simulating...')
        t0 = time()
        for _ in range(N_steps):
            auto.step()
        print('Simulation took : ', time()-t0)
        mass_f = auto.mass() #  (B,3)

        dead_mask = (mass_f.max(dim=1).values < threshold) # (B,) True if dead
        print('Adjusting...')
        for key,mid_param in mid_params.items():
            if(key!='k_size'):
                p_d[key][dead_mask] = mid_param[dead_mask] # Move dead point
                p_a[key][~dead_mask] = mid_param[~dead_mask] # Move alive point
                
                t_crit[dead_mask] += 0.5**(i+2) # Move t_crit for dead
                t_crit[~dead_mask] -= 0.5**(i+2) # Move t_crit for alive

    mid_params = mean_params(p_d,p_a)

    return t_crit, mid_params


def mean_params(p1,p2):
    assert p1['k_size']==p2['k_size'], 'Kernel sizes must match'

    new_p = {'k_size' : p1['k_size']}
    for key in p1:
        if(key!='k_size'):
            new_p[key] = (p2[key]+p1[key])/2.
    
    return new_p


def hash_dict(d):
    """
        Produces hash for dictionary parameters.
    """
    # Convert the dictionary into a sorted string
    d_str = str(sorted(d.items())).encode('utf-8')
    
    # Use SHA-256 to hash the string and return the hexdigest
    return hashlib.sha256(d_str).hexdigest()[:8]

def param_batch_to_list(b_params,new_batch_size=1,squeeze=True):
    """
        Separates a batched parameter dictionary into a list of batched parameters, with a new batch_size.
        NOTE : Last element in the list might have a batch_size smaller than new_batch_size.

        Args:
            b_params : batched parameters (dict)
            new_batch_size : new batch size (int)
            squeeze : if True, removes batch_size dimension if new_batch_size=1
        Returns:
            list of batched parameters list[dict]
    """
    if(new_batch_size>=b_params['mu'].shape[0] and not squeeze):
        return [b_params]

    batch_size = b_params['mu'].shape[0]
    param_list = []
    if(squeeze and new_batch_size!=1):
        squeeze=False

    for i in range(ceil(batch_size/new_batch_size)):
        param_list.append({'k_size' : b_params['k_size']})
        for key in b_params:
            if(key!='k_size'):
                # Cut to (new_batch_size,*) and add to list
                param_list[-1][key] = (b_params[key][i*new_batch_size:min((i+1)*new_batch_size,batch_size)])
                if(squeeze):
                    param_list[-1][key] = param_list[-1][key].squeeze(0)
    
    return param_list

def expand_batch(param,tar_batch):
    """
        Expands parameters of batch_size=1 to a target batch size.
        Args:
            param : batch of parameters (dict)
            tar_batch : target batch size (int)
        Returns:
            expanded batch of parameters (dict)
    """
    batch_size = param['mu'].shape[0]
    assert batch_size==1, 'original batch size must be 1'

    new_param = {'k_size' : param['k_size']}
    for key in param:
        if(key!='k_size'):
            n_d = len(param[key].shape)-1
            new_param[key] = param[key].repeat(tar_batch,*([1]*n_d))
    
    return new_param

def save_param(batch_folder,indiv_folder,params):
    """
        Saves parameter both in batch and individually.

        Returns : path to batch params.
    """
    name = hash_dict(params)
    batch_size = params['mu'].shape[0]

    if(batch_size>1):
        f = open(os.path.join(batch_folder,name+'.pk'), "wb") 
        pk.dump(params,f) # Save the resulting parameter batch
        f.close() 

    mid_params_list = param_batch_to_list(params) # Unbatched list of dicts
    for j in range(len(mid_params_list)):
        f = open(os.path.join(indiv_folder,name[:-3]+f'{j:02d}'+'.pk'), "wb") 
        pk.dump(mid_params_list[j],f)
        f.close()
    
    return os.path.join(batch_folder,name+'.pk') 

def save_rand(dir,batch_size,num,param_generator,device='cpu'):
    dir_b = os.path.join(dir,'batch')
    dir_i = os.path.join(dir,'individual')

    os.makedirs(dir_b,exist_ok=True)
    os.makedirs(dir_i,exist_ok=True)
    for i in range(num):
        params = param_generator(batch_size,device=device)
        save_param(dir_b,dir_i,params)
