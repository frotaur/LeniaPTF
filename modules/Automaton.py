import torch,torch.nn,torch.nn.functional as F
import numpy as np
from torchenhanced import DevModule
from .utils.noise_gen import perlin,perlin_fractal
from .utils.main_utils import gen_batch_params,gen_params

class Automaton(DevModule) :
    """
        Class that internalizes the rules and evolution of 
        the cellular automaton at hand. It has a step function
        that makes one timestep of the evolution. By convention,
        and to keep in sync with pygame, the world tensor has shape
        (W,H,3). It contains float values between 0 and 1, which
        are (automatically) mapped to 0 255 when returning output, 
        and describes how the world is 'seen' by an observer.

        Parameters :
        size : 2-uple (W,H)
            Shape of the CA world
        device : str
    """

    def __init__(self,size, device='cpu'):
        super().__init__()
        self.w, self.h  = size
        self.size= size
        # This self._worldmap should be changed in the step function.
        # It should contains floats from 0 to 1 of RGB values.
        self._worldmap = np.random.uniform(( self.w,self.h,3))
        self.to(device)
        

    

    def step(self):
        return NotImplementedError('Please subclass "Automaton" class, and define self.step')
    
    @property
    def worldmap(self):
        return (255*self._worldmap).astype(dtype=np.uint8)

class LeniaMC(Automaton):
    """
        Generalized Lenia-like automaton.

        Args :
        size : tuple of ints, size of the automaton
        dt : time-step used when computing the evolution of the automaton
        params : dict of tensors containing the parameters.
            keys-values : 
            'k_size' : odd int, size of kernel used for computations
            'mu' : (3,3) tensor, mean of growth functions
            'sigma' : (3,3) tensor, standard deviation of the growth functions
            'beta' :  (3,3, # of rings) float, max of the kernel rings 
            'mu_k' : (3,3, # of rings) [0,1.], location of the kernel rings
            'sigma_k' : (3,3, # of rings) float, standard deviation of the kernel rings
            'weights' : (3,3) float, weights for the growth weighted sum
        device : str, device 
    """
    
    def __init__(self, size, dt, params=None, state_init = None, device='cpu' ):
        super().__init__(size, device)
        if(params is None):
            params = gen_params(device)
        # 0,1,2,3 of the first dimension are the N,W,S,E directions
        self.k_size = params['k_size'] # kernel sizes (same for all)

        self.register_buffer('state',torch.rand((3,self.h,self.w)))

        if(state_init is None):
            self.set_init_fractal()
        else:
            self.state = state_init.to(self.device)

        self.dt = dt
        # Buffer for all parameters since we do not require_grad for them :

        self.register_buffer('mu', params['mu']) # mean of the growth functions (3,3)
        self.register_buffer('sigma', params['sigma']) # standard deviation of the growths functions (3,3)
        self.register_buffer('beta',params['beta']) # max of the kernel rings (3,3, # of rings)
        self.register_buffer('mu_k',params['mu_k'])# mean of the kernel gaussians (3,3, # of rings)
        self.register_buffer('sigma_k',params['sigma_k'])# standard deviation of the kernel gaussians (3,3, # of rings)
        self.register_buffer('weights',params['weights']) # weigths for the growth weighted sum (3,3)

        self.norm_weights()
        self.register_buffer('kernel',torch.zeros((self.k_size,self.k_size)))
        self.kernel = self.compute_kernel() # (3,3,h, w)

    def norm_weights(self):
        # Normalizing the weights
        N = self.weights.sum(dim=0, keepdim = True)
        self.weights = torch.where(N > 1.e-6, self.weights/N, 0)
    
    def update_params(self, params):
        """
            Updates the parameters of the automaton.
        """
        # Add kernel size
        self.k_size = params['k_size'] # kernel sizes (same for all)
        # Possibly add class 'PArams' to hold all parameter and implement binary operators
        self.mu = params['mu'] # mean of the growth functions (3,3)
        self.sigma = params['sigma'] # standard deviation of the growths functions (3,3)
        self.beta = params['beta']
        self.mu_k = params['mu_k']
        self.sigma_k = params['sigma_k']
        self.weights = params['weights']
        self.norm_weights()

        self.kernel = self.compute_kernel() # (3,3,h, w)


    def get_params(self):
        """
            Get the parameter dictionary which defines the automaton
        """
        # Add output of kernel size
        params = dict(mu = self.mu, sigma = self.sigma, beta = self.beta,
                       mu_k = self.mu_k, sigma_k = self.sigma_k, weights = self.weights)
        
        return params


    def set_init_fractal(self):
        """
            Sets the initial state of the automaton using perlin noise
        """
        self.state = perlin_fractal((1,self.h,self.w),int(self.k_size*1.5),device=self.device,black_prop=0.25,persistence=0.4)[0]    
    
    def set_init_perlin(self,wavelength=None):
        if(not wavelength):
            wavelength = self.k_size
        self.state = perlin((1,self.h,self.w),[wavelength]*2,device=self.device,black_prop=0.25)[0]
    
    def kernel_slice(self, r): # r : (k_size,k_size)
        """
            Given a distance matrix r, computes the kernel of the automaton.

            Args :
            r : (k_size,k_size), value of the radius for each location around the center of the kernel
        """
        #K = torch.ones((3,3,self.k_size, self.k_size))
        r = r[None, None, None] #(1, 1, 1, k_size, k_size)
        r = r.expand(3,3,self.mu_k[0][0].size()[0],-1,-1) #(3,3,#of rings,k_size,k_size)

        mu_k = self.mu_k[:,:,:, None, None]
        sigma_k = self.sigma_k[:,:,:, None, None]

        # K = torch.exp(-((r-mu_k)/2)**2/sigma_k) #(3,3,#of rings,k_size,k_size)
        K = torch.exp(-((r-mu_k)/sigma_k)**2/2) 
        #print(K.shape)

        beta = self.beta[:,:,:, None, None]

        K = torch.sum(beta*K, dim = 2)

        
        return K #(3,3,k_size, k_size)
    
    def compute_kernel(self):
        """
            Computes the kernel given the parameters.
        """
        xyrange = torch.arange(-1, 1+0.00001, 2/(self.k_size-1)).to(self.device)
        X,Y = torch.meshgrid(xyrange, xyrange,indexing='ij')
        r = torch.sqrt(X**2+Y**2)

        K = self.kernel_slice(r) #(3,3,k_size,k_size)

        # Normalize the kernel
        summed = torch.sum(K, dim = (2,3), keepdim=True) #(3,3,1,1)

        # Avoid divisions by 0
        summed = torch.where(summed<1e-6,1,summed)
        K /= summed

        return K #(3,3,k,k)
    
    def growth(self, u): # u:(3,3,H,W)
        """
            Computes the growth of the automaton given the concentration u.

            Args :
            u : (3,3,H,W) tensor of concentrations.
        """

        # Possibly in the future add other growth function using bump instead of guassian
        mu = self.mu[:,:, None, None]
        sigma = self.sigma[:,:,None,None]
        mu = mu.expand(-1,-1, self.h, self.w)
        sigma = sigma.expand(-1,-1, self.h, self.w)

        return 2*torch.exp(-((u-mu)**2/(sigma)**2)/2)-1 #(3,3,H,W)


    def step(self):
        """
            Steps the automaton state by one iteration.
        """
        # Shenanigans to make all the convolutions at once.
        kernel_eff = self.kernel.reshape([9,1,self.k_size,self.k_size])#(9,1,k,k)
        # kernel_eff = torch.zeros_like(kernel_eff)
        # kernel_eff[]

        U = F.pad(self.state[None], [(self.k_size-1)//2]*4, mode = 'circular') # (1,3,H+pad,W+pad)
        U = F.conv2d(U, kernel_eff, groups=3).squeeze(0) #(9,H,W)
        U = U.reshape(3,3,self.h,self.w)

        assert (self.h,self.w) == (self.state.shape[1], self.state.shape[2])
        # Maybe change to weighted sum with the weights being also parameters  (done)
        # factor = torch.eye(3)[...,None,None].to(self.device) #(3,3)
 
        weights = self.weights [...,None, None]
        weights = weights.expand(-1, -1, self.h,self.w) # 

        # print((U*weights).sum(dim=(2,3)))
        #print('Selfgrowth shape ', self.growth(U).shape)
        #print((self.growth(U)*weights).sum(dim=(2,3)))
        dx = (self.growth(U)*weights).sum(dim=0) #(3,H,W)
        
        #showTens(dx.cpu())

        self.state = torch.clamp( self.state + self.dt*dx, 0, 1) 

    def evolve_state(self,state):
        kernel_eff = self.kernel.reshape([9,1,self.k_size,self.k_size])#(9,1,k,k)
        # kernel_eff = torch.zeros_like(kernel_eff)
        # kernel_eff[]
        U = F.pad(self.state[None], [(self.k_size-1)//2]*4, mode = 'circular')
        U = F.conv2d(U, kernel_eff, groups=3).squeeze(0) #(9,H,W)
        U = U.reshape(3,3,self.h,self.w)

        assert (self.h,self.w) == (state.shape[1], state.shape[2])
        # Maybe change to weighted sum with the weights being also parameters  (done)
        # factor = torch.eye(3)[...,None,None].to(self.device) #(3,3)
        N = self.weights.sum(dim=0, keepdim = True)
        # print(N)
        weights = torch.where(N > 1.e-6, self.weights/N, 0)
        # print(weights)
        weights = weights [...,None, None]
        weights = weights.expand(-1, -1, self.h,self.w)

        # print((U*weights).sum(dim=(2,3)))
        #print('Selfgrowth shape ', self.growth(U).shape)
        #print((self.growth(U)*weights).sum(dim=(2,3)))
        dx = (self.growth(U)*weights).sum(dim=0) #(3,H,W)
        dx.to(self.device)
        #showTens(dx.cpu())

        return torch.clamp(state + self.dt*dx, 0, 1)
    
    def draw(self):
        """
            Draws the worldmap from state.
            Separate from step so that we can freeze time,
            but still 'paint' the state and get feedback.
        """
        
        toshow= self.state.permute((2,1,0)) #(W,H,3)

        self._worldmap= toshow.cpu().numpy()   
        
    def mass(self):
        """
            Computes average 'mass' of the automaton for each channel

            returns :
            mass : (3,) tensor, mass of each channel
        """

        return self.state.mean(dim=(1,2)) # (3, ) mean mass for each color
       
class BatchLeniaMC(DevModule):
    """
        Batched Multi-channel lenia, to run batch_size worlds in parallel !
        Does not support live drawing in pygame, mayble will later.
    """
    def __init__(self, size, dt, num_channels=3, params=None, state_init = None, device='cpu' ):
        """
            Initializes automaton.  

            Args :
                size : (B,H,W) of ints, size of the automaton and number of batches
                dt : time-step used when computing the evolution of the automaton
                num_channels : int, number of channels (C) in the automaton
                params : dict of tensors containing the parameters. If none, generates randomly
                    keys-values : 
                    'k_size' : odd int, size of kernel used for computations
                    'mu' : (B,C,C) tensor, mean of growth functions
                    'sigma' : (B,C,C) tensor, standard deviation of the growth functions
                    'beta' :  (B,C,C, # of rings) float, max of the kernel rings 
                    'mu_k' : (B,C,C, # of rings) [0,1.], location of the kernel rings
                    'sigma_k' : (B,C,C, # of rings) float, standard deviation of the kernel rings
                    'weights' : (B,C,C) float, weights for the growth weighted sum
                device : str, device 
        """
        super().__init__()
        self.to(device)

        self.batch= size[0]
        self.h, self.w  = size[1:]
        self.C = num_channels
        # 0,1,2,3 of the first dimension are the N,W,S,E directions
        if(params is None):
            params = gen_batch_params(self.batch,device,num_channels=self.C)
        self.k_size = params['k_size'] # kernel sizes (same for all)

        self.register_buffer('state',torch.rand((self.batch,self.C,self.h,self.w)))

        if(state_init is None):
            self.set_init_fractal()
        else:
            self.state = state_init.to(self.device)

        self.dt = dt

        # Buffer for all parameters since we do not require_grad for them :
        self.register_buffer('mu', params['mu']) # mean of the growth functions (3,3)
        self.register_buffer('sigma', params['sigma']) # standard deviation of the growths functions (3,3)
        self.register_buffer('beta',params['beta']) # max of the kernel rings (3,3, # of rings)
        self.register_buffer('mu_k',params['mu_k'])# mean of the kernel gaussians (3,3, # of rings)
        self.register_buffer('sigma_k',params['sigma_k'])# standard deviation of the kernel gaussians (3,3, # of rings)
        self.register_buffer('weights',params['weights']) # raw weigths for the growth weighted sum (3,3)

        self.norm_weights()
        self.register_buffer('kernel',torch.zeros((self.k_size,self.k_size)))
        self.kernel = self.compute_kernel() # (3,3,h, w)


    def update_params(self, params):
        """
            Updates some or all parameters of the automaton. Changes batch size to match one of provided params (take mu as reference)
        """
        self.mu = params.get('mu',self.mu) # mean of the growth functions (C,C)
        self.sigma = params.get('sigma',self.sigma) # standard deviation of the growths functions (C,C)
        self.beta = params.get('beta',self.beta)
        self.mu_k = params.get('mu_k',self.mu_k)
        self.sigma_k = params.get('sigma_k',self.sigma_k)
        self.weights = params.get('weights',self.weights)
        self.k_size = params.get('k_size',self.k_size) # kernel sizes (same for all)

        self.norm_weights()

        self.batch = self.mu.shape[0] # update batch size
        self.kernel = self.compute_kernel() # (B,C,C,h, w)

    
    def norm_weights(self):
        # Normalizing the weights
        N = self.weights.sum(dim=1, keepdim = True) # (B,3,3)
        self.weights = torch.where(N > 1.e-6, self.weights/N, 0)

    def get_params(self):
        """
            Get the parameter dictionary which defines the automaton
        """
        params = dict(k_size = self.k_size,mu = self.mu, sigma = self.sigma, beta = self.beta,
                       mu_k = self.mu_k, sigma_k = self.sigma_k, weights = self.weights)
        
        return params

    def set_init_fractal(self):
        """
            Sets the initial state of the automaton using perlin noise
        """
        self.state = perlin_fractal((self.batch,self.h,self.w),int(self.k_size*1.5),
                                    device=self.device,black_prop=0.25,num_channels=self.C,persistence=0.4) 
    
    def set_init_perlin(self,wavelength=None):
        if(not wavelength):
            wavelength = self.k_size
        self.state = perlin((self.batch,self.h,self.w),[wavelength]*2,
                            device=self.device,num_channels=self.C,black_prop=0.25)
        
    def kernel_slice(self, r): # r : (k_size,k_size)
        """
            Given a distance matrix r, computes the kernel of the automaton.

            Args :
            r : (k_size,k_size), value of the radius for each location around the center of the kernel
        """

        r = r[None, None, None,None] #(1,1, 1, 1, k_size, k_size)
        r = r.expand(self.batch,self.C,self.C,self.mu_k[0][0].size()[0],-1,-1) #(B,C,C,#of rings,k_size,k_size)

        mu_k = self.mu_k[..., None, None] # (B,C,C,#of rings,1,1)
        sigma_k = self.sigma_k[..., None, None]# (B,C,C,#of rings,1,1)

        K = torch.exp(-((r-mu_k)/sigma_k)**2/2) #(B,C,C,#of rings,k_size,k_size)
        #print(K.shape)

        beta = self.beta[..., None, None] # (B,C,C,#of rings,1,1)

        K = torch.sum(beta*K, dim = 3)

        
        return K #(B,C,C,k_size, k_size)
    
    def compute_kernel(self):
        """
            Computes the kernel given the parameters.
        """
        xyrange = torch.arange(-1, 1+0.00001, 2/(self.k_size-1)).to(self.device)
        X,Y = torch.meshgrid(xyrange, xyrange,indexing='ij')
        r = torch.sqrt(X**2+Y**2)

        K = self.kernel_slice(r) #(B,C,C,k_size,k_size)

        # Normalize the kernel
        summed = torch.sum(K, dim = (-1,-2), keepdim=True) #(B,C,C,1,1)

        # Avoid divisions by 0
        summed = torch.where(summed<1e-6,1,summed)
        K /= summed

        return K #(B,C,C,k,k)
    
    def growth(self, u): # u:(B,C,C,H,W)
        """
            Computes the growth of the automaton given the concentration u.

            Args :
            u : (B,C,C,H,W) tensor of concentrations.
        """

        # Possibly in the future add other growth function using bump instead of guassian
        mu = self.mu[..., None, None] # (B,C,C,1,1)
        sigma = self.sigma[...,None,None] # (B,C,C,1,1)
        mu = mu.expand(-1,-1,-1, self.h, self.w) # (B,C,C,H,W)
        sigma = sigma.expand(-1,-1,-1, self.h, self.w) # (B,C,C,H,W)

        return 2*torch.exp(-((u-mu)**2/(sigma)**2)/2)-1 #(B,C,C,H,W)


    def step(self):
        """
            Steps the automaton state by one iteration.

            Args :
            discrete_g : 2-uple of floats, min and max values, when using 'discrete' growth.
            If None, will use the normal growth function.
        """
        # Shenanigans to make all the convolutions at once.
        kernel_eff = self.kernel.reshape([self.batch*self.C*self.C,1,self.k_size,self.k_size])#(B*C^2,1,k,k)

        U = self.state.reshape(1,self.batch*self.C,self.h,self.w) # (1,B*C,H,W)
        U = F.pad(U, [(self.k_size-1)//2]*4, mode = 'circular') # (1,B*C,H+pad,W+pad)
        
        U = F.conv2d(U, kernel_eff, groups=self.C*self.batch).squeeze(1) #(B*C^2,1,H,W) squeeze to (B*9,H,W)
        U = U.reshape(self.batch,self.C,self.C,self.h,self.w) # (B,C,C,H,W)

        assert (self.h,self.w) == (self.state.shape[2], self.state.shape[3])

        weights = self.weights[...,None, None] # (B,C,C,1,1)
        weights = weights.expand(-1,-1, -1, self.h,self.w) # (B,C,C,H,W)

        dx = (self.growth(U)*weights).sum(dim=1) #(B,C,H,W)

        self.state = torch.clamp(self.state + self.dt*dx, 0, 1)     

    def discrete_step(self):
        """
            Steps the automaton state by one iteration, in the discrete case
        """
        # Shenanigans to make all the convolutions at once.
        kernel_eff = self.kernel.reshape([self.batch*self.C*self.C,1,self.k_size,self.k_size])#(B*C^2,1,k,k)

        U = self.state.reshape(1,self.batch*self.C,self.h,self.w) # (1,B*C,H,W)
        U = F.pad(U, [(self.k_size-1)//2]*4, mode = 'circular') # (1,B*C,H+pad,W+pad)
        
        U = F.conv2d(U, kernel_eff, groups=self.C*self.batch).squeeze(1) #(B*C^2,1,H,W) squeeze to (B*9,H,W)
        U = U.reshape(self.batch,self.C,self.C,self.h,self.w) # (B,C,C,H,W)

        assert (self.h,self.w) == (self.state.shape[2], self.state.shape[3])

        dx = ((U > discrete_g[0]) & (U < discrete_g[1])).float()
        dx = dx * 2 - 1

        self.state = torch.clamp(self.state + self.dt*dx, 0, 1)  

    def mass(self):
        """
            Computes average 'mass' of the automaton for each channel

            returns :
            mass : (B,C) tensor, mass of each channel
        """

        return self.state.mean(dim=(-1,-2)) # (B,C) mean mass for each color

    def draw(self):
        """
            Draws the worldmap from state.
            Separate from step so that we can freeze time,
            but still 'paint' the state and get feedback.
        """
        assert self.state.shape[0] == 1, "Batch size must be 1 to draw"
        toshow= self.state[0].permute((2,1,0)) #(W,H,C)

        if(self.C==1):
            toshow = toshow.expand(-1,-1,3)
        elif(self.C==2):
            toshow = torch.cat([toshow,torch.zeros_like(toshow)],dim=-1)
        else :
            toshow = toshow[:,:,:3]
    
        self._worldmap= toshow.cpu().numpy()   
    
        
    @property
    def worldmap(self):
        return (255*self._worldmap).astype(dtype=np.uint8)

class DiscreteLenia(DevModule):
    """
        Discrete Batched Single-channel Lenia, to run batch_size worlds in parallel !
    """
    def __init__(self, size, discretization:int, params=None, state_init = None, device='cpu' ):
        """
            Initializes automaton.  

            Args :
                size : (B,H,W) of ints, size of the automaton and number of batches
                discretization : int, number of discrete values for the automaton
                params : dict of tensors containing the parameters. If none, generates randomly
                    keys-values : 
                    'k_size' : odd int, size of kernel used for computations
                    'mu' : (B,) tensor, mean of growth functions
                    'sigma' : (B,) tensor, standard deviation of the growth functions
                    'mu_k' : (B,) [0,1.], location of the kernel rings (NOT USED FOR NOW)
                    'sigma_k' : (B,) float, standard deviation of the kernel rings (NOT USED FOR NOW)
                    'device' : str, device 
        """
        super().__init__()
        self.to(device)

        self.batch= size[0]
        self.h, self.w  = size[1:]
        # 0,1,2,3 of the first dimension are the N,W,S,E directions
        if(params is None):
            params = self.gen_batch_params(device)
    
        self.k_size = params['k_size'] # kernel sizes (same for all)
        self.discri = discretization

        self.register_buffer('state',torch.randint(low=0,high=discretization+1,size=(self.batch,1,self.h,self.w)))

        if(state_init is None):
            self.set_init_fractal()
        else:
            self.state = state_init.to(self.device,dtype=torch.int)

        # Buffer for all parameters since we do not require_grad for them :
        self.register_buffer('mu', params['mu']) # mean of the growth functions (3,3)
        self.register_buffer('sigma', params['sigma']) # standard deviation of the growths functions (3,3)
        self.register_buffer('mu_k',params['mu_k'])# mean of the kernel gaussians (3,3, # of rings)
        self.register_buffer('sigma_k',params['sigma_k'])# standard deviation of the kernel gaussians (3,3, # of rings)

        self.register_buffer('kernel',torch.zeros((self.k_size,self.k_size)))
        self.update_params(params)

    def gen_batch_params(self,device):
        """ Generates batch parameters."""
        mu = 0.15*(torch.ones((self.batch,), device=device))
        sigma = 0.015*(torch.ones_like(mu))
            

        params = {
            'k_size' : 27, 
            'mu':  mu ,
            'sigma' : sigma,
            'mu_k' : torch.full((self.batch,), fill_value=0.5, device=device),
            'sigma_k' : torch.full((self.batch,),fill_value=0.15, device=device),
        }
        
        return params

    def update_params(self, params):
        """
            Updates some or all parameters of the automaton. Changes batch size to match one of provided params (take mu as reference)
        """
        self.mu = params.get('mu',self.mu) # mean of the growth functions (C,C)
        self.sigma = params.get('sigma',self.sigma) # standard deviation of the growths functions (C,C)
        self.mu_k = params.get('mu_k',self.mu_k)
        self.sigma_k = params.get('sigma_k',self.sigma_k)
        self.k_size = params.get('k_size',self.k_size) # kernel sizes (same for all)


        self.batch = self.mu.shape[0] # update batch size
        self.update_kern_growth()
    
    def update_kern_growth(self):
        self.kernel, max_kern_activ = self.compute_kernel() # (B,1,1,h, w)
        self.growths = self.compute_growth(max_kern_activ)[:,:,None,None,None,None] # (B,2,1,1,1,1) of min and max growths (comparable direclty with state)

    def compute_growth(self,max_kernel_activation):
        """
            Create growth range given parameters
        """

        growth_x_axes =  [torch.linspace(0, 1, max_kern.item(),device=self.device) for max_kern in max_kernel_activation] # mapping from activation to [0,1], shape (max_kern,)*B
        g_tensor = [] # (B,2) tensor of min and max growth values
        for i,growth_axis in enumerate(growth_x_axes):
            growth_lookup = self.gaussian_func(growth_axis, m=self.mu[i], s=self.sigma[i]) # growth value per activation (max_kern,)

            growth_lookup = self.discretize(growth_lookup, div=1, mult=True) # discretize the growth values to [0, discri]

            growth_support = growth_lookup.nonzero()[:,0] # true if growth is nonzero

            if growth_support.any():
                arg_min, arg_max = torch.min(growth_support).item(), torch.max(growth_support).item()
                g = torch.tensor([max(0,arg_min-1), min(arg_max+1,growth_lookup.shape[0]-1)], dtype=torch.int, device=self.device)
            else:
                g = torch.tensor([0, 0], dtype=torch.int, device=self.device)

            g_tensor.append(g)
        # print('DA G TENSOR : ', torch.stack(g_tensor,dim=0))
        
        return torch.stack(g_tensor,dim=0) # (B,2)
    
    def get_params(self):
        """
            Get the parameter dictionary which defines the automaton
        """
        params = dict(k_size = self.k_size,mu = self.mu, sigma = self.sigma, beta = self.beta,
                       mu_k = self.mu_k, sigma_k = self.sigma_k, weights = self.weights)
        
        return params

    def set_init_fractal(self):
        """
            Sets the initial state of the automaton using perlin noise
        """
        perlin = perlin_fractal((self.batch,self.h,self.w),int(self.k_size*1.5),
                                    device=self.device,black_prop=0.25,num_channels=1,persistence=0.4) 

        self.state = (perlin*self.discri).round().clamp(0,self.discri)


    def set_init_perlin(self,wavelength=None, square_size=None):
        if(not wavelength):
            wavelength = self.k_size
        perlino = perlin((self.batch,self.h,self.w),[wavelength]*2,
                            device=self.device,num_channels=1,black_prop=0.25)

        if(square_size):
            masku = torch.zeros_like(perlino)
            masku[:,:,self.h//2-square_size//2:self.h//2+square_size//2,self.w//2-square_size//2:self.w//2+square_size//2] = 1
            perlino = perlino*masku
        self.state = (perlino*self.discri).round().clamp(0,self.discri)
    
    @staticmethod
    def gaussian_func(x, m, s, h=1):
        def safe_divide(x, y, eps=1e-10):
            return x / (y + eps)
        return torch.exp(-safe_divide((x - m), s)**2 / 2) * h
    
    @staticmethod
    def discretize(tensor, div=1, mult=True):
        if mult:
            return torch.round(tensor * div).to(dtype=torch.int)
        else:
            return tensor.to(dtype=torch.int)

    def compute_kernel(self):
        """
            Get the kernel in the case k=1
        """
        # calculate distance from origin
        kernel_sizes = [self.k_size, self.k_size] # (k_y, k_x)
        kernel_radius = (self.k_size-1)//2

        kernel_mids = [size // 2 for size in kernel_sizes] # (mid_y, mid_x)

        ranges = [slice(0 - mid, size - mid) for size, mid in zip(kernel_sizes, kernel_mids)] # y range, x range

        space = np.asarray(np.mgrid[ranges], dtype=float)  # [2, k_x,k_y]. space[:,x,y] = [x,y]
        distance = np.linalg.norm(space, axis=0)  # [k_x,k_y]. distance[x,y] = sqrt(x^2 + y^2)

        # calculate kernel K
        distance_scaled = distance / kernel_radius  # [xyz]
        
        distance_scaled = torch.tensor(distance_scaled).to(self.device) # (k_x,k_y) tensor of r_distances
        distance_scaled = distance_scaled[None].expand(self.batch,-1,-1) # (B,k_x,k_y) tensor of r_distances

        kernel = self.gaussian_func(distance_scaled,m=self.mu_k[:,None,None], s=self.sigma_k[:,None,None])  # (B,k_x,k_y) tensor of kernel values
        kernel = self.discretize(kernel, self.discri, mult=True) # (B,k_x,k_y) tensor of discretized kernel values

        kernel_sum = torch.sum(kernel,dim=(-1,-2))  # [ B, ] 

        kernel_max_activation = self.discri * kernel_sum # [ B, ] max activations

        kernel = kernel.reshape(self.batch,1, 1, self.k_size, self.k_size)  # (B,1,1,k_x,k_y) tensor of kernel values

        # showTens(kernel.float())
        return kernel.float() , kernel_max_activation

    def step(self):
        """
            Steps the automaton state by one iteration.

            Args :
            discrete_g : 2-uple of floats, min and max values, when using 'discrete' growth.
            If None, will use the normal growth function.
        """
        # Shenanigans to make all the convolutions at once.
        kernel_eff = self.kernel.reshape([self.batch,1,self.k_size,self.k_size])#(B,1,k,k)

        U = self.state.reshape(1,self.batch,self.h,self.w) # (1,B,H,W)
        U = F.pad(U, [(self.k_size-1)//2]*4, mode = 'circular') # (1,B,H+pad,W+pad)
        
        U = F.conv2d(U, kernel_eff, groups=self.batch).squeeze(1) #(B*1^2,1,H,W) squeeze to (B*1,H,W)
        U = U.reshape(self.batch,1,1,self.h,self.w) # (B,1,1,H,W)

        assert (self.h,self.w) == (self.state.shape[2], self.state.shape[3])
        dx = ((U > self.growths[:,0]) & (U < self.growths[:,1])).to(dtype=torch.int)
        dx = (dx * 2 - 1).sum(dim=1) # -1 if not grown, 1 if grown (B,1,1,H,W) -> (B,1,H,W)

        self.state = torch.clamp(torch.round(self.state + dx), 0, self.discri)     

    def mass(self):
        """
            Computes average 'mass' of the automaton for each channel

            returns :
            mass : (B,C) tensor, mass of each channel
        """

        return (self.state).mean(dim=(-1,-2))/self.discri # (B,1) normalized mean mass for each color

    def draw(self):
        """
            Draws the worldmap from state.
            Separate from step so that we can freeze time,
            but still 'paint' the state and get feedback.
        """
        assert self.state.shape[0] == 1, "Batch size must be 1 to draw"
        toshow= self.state[0].permute((2,1,0)) #(W,H,C)

        toshow = toshow.expand(-1,-1,3).to(torch.float)
        toshow = toshow/self.discri # normalize to [0,1]

    
        self._worldmap= toshow.cpu().numpy()   
    
        
    @property
    def worldmap(self):
        return (255*self._worldmap).astype(dtype=np.uint8)
