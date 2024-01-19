import torch, hashlib
from time import time



@torch.no_grad()
def phase_finder(W, H, dt, N_steps, params_generator, threshold,  device='cpu'):
    """
        Finds a set of parameter of dead automaton, and a set of parameters of an alive automaton.

        Args:
            W, H : width and height of the automaton 
            dt : time step
            N_steps : number of simulation steps before checking phase
            params_generator : function which returns a set of parameters
            threshold : threshold below which we say we have found a dead config
            device : device on which to run the automaton

    """
    from ..Automaton import LeniaMC
    stop_d = True
    stop_a = True

    params = params_generator(device)
    auto = LeniaMC((W,H), dt, params, device=device)
    auto.to(device)

    while (stop_a or stop_d):
        # Initialize the automaton
        t0 = time()
        for _ in range(N_steps):
            auto.step()
        print('Simulation took : ', time()-t0)	
        mass_f = auto.mass()
        if ((max(mass_f) < threshold) and stop_d): 
            # f = open("multi_sort/dies/seed="+str(torch.seed()), "wb") 
            # pk.dump(params,f)
            # f.close()
            params_d = params
            stop_d=False
            print("Found dead")  
        elif ((max(mass_f) > threshold) and stop_a):
            # f = open("multi_sort/lives/seed="+str(torch.seed()), "wb") 
            # pk.dump(params,f)
            # f.close()
            params_e = params
            stop_a=False
            print("Found alive")

        params = params_generator(device)
        auto.update_params(params)
        auto.set_init_perlin()
    return params_d, params_e
       
@torch.no_grad()
def interest_finder(W,H, dt, N_steps, params_d, params_a, refinement, threshold, device):
    """
        By dichotomy, finds the parameters of an interesting automaton. By interesting, here
        we mean a set of parameters which lies at the transition between an asymptotically dead
        and an asymptotically alive automaton.

        Args :
            W, H : width and height of the automaton 
            dt : time step
            N_steps : number of steps the automaton goes through for each set of parameters
            device : device on which to run the automaton
            params_d : parameters of a dead automaton   (dict)
            params_a : parameters of an alive automaton  (dict)
            refinement : number of iterations of dichotomy
    """
    from ..Automaton import LeniaMC
    p1 = params_d
    p2 = params_a
    t_crit = 0.5

    for i in range(refinement):

        params = {
            'k_size' : 25, 
            'mu':  (p1['mu']+p2['mu'])/2 ,
            'sigma' : (p1['sigma']+p2['sigma'])/2 ,
            'beta' : (p1['beta']+p2['beta'])/2,
            'mu_k' : (p1['mu_k']+p2['mu_k'])/2,
            'sigma_k' : (p1['sigma_k']+p2['sigma_k'])/2,
            'weights' : (p1['weights']+p2['weights'])/2 # element i, j represents contribution from channel i to channel j
        }

        auto = LeniaMC((W,H), dt, params ,device=device)
        auto.to(device)

        for _ in range(N_steps):
            # Step to get to the 'final' state
            auto.step()
        
        if (max(auto.mass()) < threshold): # Also put this value as parameter, and the same in both ifs
            p1 = params
            t_crit += 0.5**(i+2)
        elif(max(auto.mass()) > threshold):
            p2 = params
            t_crit -= 0.5**(i+2)
    #print("Interest found !")

    return t_crit


def hash_dict(d):
    """
        Produces hash for dictionary parameters.
    """
    # Convert the dictionary into a sorted string
    d_str = str(sorted(d.items())).encode('utf-8')
    
    # Use SHA-256 to hash the string and return the hexdigest
    return hashlib.sha256(d_str).hexdigest()[:8]

