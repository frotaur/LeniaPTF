import torch
from ..Lenia import BatchLeniaMC
from time import time
import copy, os, shutil, math
from tqdm import tqdm
from math import ceil
from .leniaparams import LeniaParams, BatchParams

@torch.no_grad()
def phase_finder(size, dt, N_steps, batch_size, params_generator, 
                       threshold, num_channels=3,num_examples=None, use_mean=True, device='cpu') -> tuple[BatchParams]:
    """
        Finds a set of parameter of dead automaton, and a set of parameters of an alive automaton.

        Args:
            size : (H,W) height and width of world
            dt : time step
            N_steps : number of simulation steps before checking phase
            params_generator : function which returns a batch of parameters
            threshold : threshold below which we say we have found a dead config
            num_channels : number of channels in the automaton
            num_examples: Number of params of each phase to find.
                If None, generates batch_size//2 dead and batch_size//2 alive automata.
            use_mean : if True, uses mean of mass_f to determine if dead or alive, else uses max
            device : device on which to run the automaton
        
        Returns: 2-uple (dead_params,alive_params)
            dead_params : Dictionary of pameters, batch_size is num_examples
            alive_params : Dictionary of pameters, batch_size is num_examples
    """
    print('=====================================================')
    print('FINDING DEAD AND ALIVE PHASES')
    print('=====================================================')
    found_d = False
    found_a = False
    H,W = size

    auto = BatchLeniaMC((batch_size,*size), dt, num_channels=num_channels, device=device, use_fft=True)
    auto.to(device)

    if(num_examples is None):
        num_examples = int(batch_size//2)

    dead_params = None
    alive_params = None
    params = params_generator(batch_size,num_channels=num_channels,device=device)

    # dead_params.k_size = params['k_size']
    # alive_params.k_size = params['k_size']

    n_dead = 0
    n_alive = 0
    while (not(found_d and found_a)):
        # Initialize the automaton
        params = params_generator(batch_size,num_channels=num_channels,device=device) # Params for next run TODO : ACTUALLY PUT IT UP THERE
        auto.update_params(params)
        auto.set_init_perlin()

        print('Auto k_size : ', auto.k_size)
        t0 = time()
        for _ in range(N_steps):
            auto.step()

        print('Simulation took : ', time()-t0)	
        if(use_mean):
            mass_f = auto.mass().mean(dim=1) #  (B,)
        else:
            mass_f = auto.mass().max(dim=1).values #  (B,)
        
        dead_mask = mass_f < threshold # (B,) True if dead
        num_d = dead_mask.sum().item() # Number of dead examples in batch
        num_a = (~dead_mask).sum().item()
        # print('Dead masses : ', mass_f[dead_mask])
        # print('Alive masses : ', mass_f[~dead_mask])
        # print(f'Found {num_d} dead and {num_a} alive')
        dead_add = min(num_examples-n_dead,num_d) # Number of dead examples to keep to reach num_examples
        alive_add = min(num_examples-n_alive,num_a) # Number of alive examples to keep to reach num_examples

        param_d = params[dead_mask] 
        param_a = params[~dead_mask] 

        param_d = param_d[:dead_add]
        param_a = param_a[:alive_add]

        if(dead_params is None):
            dead_params = param_d #  Leniaparams
        else:
            dead_params = dead_params.cat(param_d)
        
        if(alive_params is None):
            alive_params = param_a
        else:
            alive_params = alive_params.cat(param_a)

        # for key,cur_param in params.items():
        #     if(key!='k_size'):
        #         B = cur_param.shape[0]
        #         par_size = cur_param.shape[1:]

        #         param_d = cur_param[dead_mask] # (Flattened selection)
        #         param_a = cur_param[~dead_mask] # (Flattened selection)

        #         param_d = param_d.reshape(-1,*par_size)[:dead_add] # (num_D,par_size)
        #         param_a = param_a.reshape(-1,*par_size)[:alive_add] # (num_A,par_size)

        #         if(key in dead_params):
        #             dead_params[key] = torch.cat((dead_params[key],param_d),dim=0) # (n_dead+num_D,par_size)
        #         else:
        #             dead_params[key] = param_d

        #         if(key in alive_params):
        #             alive_params[key] = torch.cat((alive_params[key],param_a),dim=0) # (n_alive+num_A,par_size)
        #         else:
        #             alive_params[key] = param_a

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
def interest_finder(size, dt, N_steps, p_dead:BatchParams, p_alive:BatchParams, refinement, threshold, 
                    num_channels=3, use_mean=True,device='cpu') -> tuple[torch.Tensor,BatchParams]:
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
            mean_number : number of times we simulate each set of parameters to get a mean mass
            use_mean : if True, uses mean of mass_f to determine if dead or alive, else uses max
            refinement : number of iterations of dichotomy
            threshold : threshold below which we say we have a dead config
            num_channels : number of channels in the automaton
        
        Returns:
            t_crit : threshold for which we have a transition between dead and alive
            mid_params : parameters of the automaton at the transition
    """
    print('=====================================================')
    print('Computing dichotomy on found phases')
    print('=====================================================')
    assert p_dead.batch_size==p_alive.batch_size, f'p_dead.batch_size ={p_dead.batch_size} and p_alive.batch_size={p_alive.batch_size} do not match'
    p_d = copy.deepcopy(p_dead)
    p_a = copy.deepcopy(p_alive)

    batch_size = p_a.batch_size

    t_crit = torch.full((batch_size,),0.5,device=device)

    auto = BatchLeniaMC((batch_size,*size), dt , num_channels=num_channels, device=device, use_fft=True)
    auto.to(device)

    # print('Ksize : ', p_d['k_size'])
    for i in tqdm(range(refinement)):
        mid_params = (p_d+p_a)*.5

        auto.update_params(mid_params)
        auto.set_init_perlin()

        # print('Simulating...')
        mass_f = 0
        # t0 = time()
        for _ in range(N_steps):
            auto.step()
        # print('Simulation took : ', time()-t0)
        mass_f = auto.mass() # (B,3)

        if(use_mean):
            dead_mask = mass_f.mean(dim=1) < threshold # (B,) True if dead
        else:
            dead_mask = (mass_f.max(dim=1).values < threshold) # (B,) True if dead
        
        # print('Adjusting...')
        # print(f'Step {i} masses : {mass_f.mean(dim=1)}')
        # print(f'Step {i} deadmask : {dead_mask}')
        p_d[dead_mask] = mid_params[dead_mask]
        p_a[~dead_mask] = mid_params[~dead_mask]
        t_crit[dead_mask] += 0.5**(i+2) # Move t_crit for dead
        t_crit[~dead_mask] -= 0.5**(i+2) # Move t_crit for alive
    # print('=====================================================')
    # # mid_params = p_a # Last push towards alive
    # print('dead at the end : ', dead_mask.sum().item()/batch_size)
    # print('=====================================================')
    return t_crit, mid_params

@torch.no_grad()
def search_transition(save_folder:str, param_generator:callable, num_points, N_steps:int=400, thresholds:tuple[float]=(0.01,0.01),
                       world_size=(100,100), dt=0.1, batch_size=20, refinement=8, num_channels=3,
                       save_random=False, use_means = (True,True), save_batch_params=False,
                        cross=False, device='cpu', **kwargs):
    """
        Runs a search given the parameters. Saves parameters both in save_folder
        and 'data/latest' folder, overwritten each time.

        Args:
        save_folder : folder where to save the parameters
        param_generator : function which returns a batch of parameters
        N_steps : number of steps the automaton goes through for each set of parameters
        thresholds : thresholds below which we say we have a dead config for search and 
        world_size : (H,W) height and width of world
        dt : time step
        batch_size : size of the batch of parameters
        num_channels : number of channels in the automaton
        save_random : if True, will save random parameters in save_folder+'_random'
        use_means : 2-uple of booleans, if True, uses mean of mass_f to determine if dead or alive, else uses max
        save_batch_params : if True, will save also the batched parameters in save_folder+'_batch'
        device : device on which to run the automaton
    """
    if(save_random):
        rand_folder = save_folder+'_random'

        if(os.path.exists(rand_folder)):
            shutil.rmtree(rand_folder)
            os.makedirs(rand_folder,exist_ok=True)
        save_rand(rand_folder,batch_size=batch_size,num=max(1,20//batch_size),
                          num_channels=num_channels,param_generator=param_generator,device=device)


    latest = os.path.join('data','latest')
    if(os.path.exists(latest)):
        shutil.rmtree(latest)
    os.makedirs(latest, exist_ok=True)
    
    if(save_batch_params):
        batch_folder = os.path.join(save_folder,'_batch')

        os.makedirs(batch_folder, exist_ok=True)
    else :
        batch_folder=None

    os.makedirs(save_folder, exist_ok=True)

    H,W = world_size

    threshold_e, threshold_i = thresholds
    mean_find, mean_search = use_means
    
    def _save_all(paramus):
        paramus.save_indiv(folder=save_folder, batch_name=False)
        if(batch_folder is not None):
            paramus.save(folder=batch_folder)
        paramus.save_indiv(folder=latest,batch_name=True)

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
                phase_finder((H,W), dt, N_steps, batch_size=batch_size,params_generator=param_generator, 
                                            threshold=threshold_e, num_channels=num_channels,num_examples=min(batch_size,num_each),
                                            use_mean=mean_find, device=device) 
            
            if(cross):
                # Compute transition point between all pairs of parameters
                for j in range(params_d.batch_size):
                    param_d = params_d[j]
                    param_d = params_d.expand(params_a.batch_size)
                    # Param_d has batch_size = 1, but will broadcast seamlessly when summing with params_a
                    _, mid_params = interest_finder((H,W), dt, N_steps, param_d, params_a, 
                                                                refinement, threshold_i, device ,num_channels=num_channels,) 
                    _save_all(mid_params)
            else:
                _, mid_params = interest_finder((H,W), dt, N_steps, params_d, params_a, 
                                                                refinement, threshold_i,use_mean=mean_search,device=device,num_channels=num_channels,)
                _save_all(mid_params)

        print(f'Total time for {num_points} : {time()-t00}')

# def param_batch_to_list(b_params,new_batch_size=1,squeeze=True):
#     """
#         Separates a batched parameter dictionary into a list of batched parameters, with a new batch_size.
#         NOTE : Last element in the list might have a batch_size smaller than new_batch_size.

#         Args:
#             b_params : batched parameters (dict)
#             new_batch_size : new batch size (int)
#             squeeze : if True, removes batch_size dimension if new_batch_size=1
#         Returns:
#             list of batched parameters list[dict]
#     """
#     if(new_batch_size>=b_params['mu'].shape[0] and not squeeze):
#         return [b_params]

#     batch_size = b_params['mu'].shape[0]
#     param_list = []
#     if(squeeze and new_batch_size!=1):
#         squeeze=False

#     for i in range(ceil(batch_size/new_batch_size)):
#         param_list.append({'k_size' : b_params['k_size']})

#         for key in b_params:
#             if(key!='k_size'):
#                 # Cut to (new_batch_size,*) and add to list
#                 param_list[-1][key] = (b_params[key][i*new_batch_size:min((i+1)*new_batch_size,batch_size)])
#                 if(squeeze):
#                     param_list[-1][key] = param_list[-1][key].squeeze(0)
    
#     return param_list

# def expand_batch(param,tar_batch):
#     """
#         Expands parameters of batch_size=1 to a target batch size.
#         Args:
#             param : batch of parameters (dict)
#             tar_batch : target batch size (int)
#         Returns:
#             expanded batch of parameters (dict)
#     """
#     batch_size = param['mu'].shape[0]
#     assert batch_size==1, 'original batch size must be 1'

#     new_param = {'k_size' : param['k_size']}
#     for key in param:
#         if(key!='k_size'):
#             n_d = len(param[key].shape)-1
#             new_param[key] = param[key].repeat(tar_batch,*([1]*n_d))
    
#     return new_param

def save_rand(folder,batch_size,num,num_channels,param_generator, batch_folder=None, device='cpu'):
    """
        Generates and saves random parameters in the given folder.

        Args :
        folder : path to folder where to save the parameters
        num : number of parameters to generate
        num_channels : number of channels in the automaton
        param_generator : function which returns a batch of parameters
        batch_folder : if provided, will save also the batched parameters
        device : device on which to generate the parameters
    """
    if(batch_folder is not None):
        os.makedirs(batch_folder,exist_ok=True)
    os.makedirs(folder,exist_ok=True)

    for _ in range(num):
        params = param_generator(batch_size,num_channels=num_channels,device=device)
        params.save_indiv(folder, batch_name=True)
        if(batch_folder is not None):
            params.save(folder=batch_folder)

