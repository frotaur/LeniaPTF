import torch,torch.nn,torch.nn.functional as F
import numpy as np
from torchenhanced import DevModule
from .utils.noise_gen import perlin,perlin_fractal
from .utils.main_utils import gen_batch_params

class BatchLeniaMC(DevModule):
    """
        Batched Multi-channel lenia, to run batch_size worlds in parallel !
        Does not support live drawing in pygame, maybe will later.
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

        if(params is None):
            # Generates random parameters
            params = gen_batch_params(self.batch,device,num_channels=self.C)

        self.k_size = params['k_size'] # kernel sizes (same for all) MUST BE ODD !!!

        self.register_buffer('state',torch.rand((self.batch,self.C,self.h,self.w)))

        if(state_init is None):
            self.set_init_fractal() # Fractal perlin init
        else:
            self.state = state_init.to(self.device) # Specific init

        self.dt = dt

        # Buffer for all parameters since we do not require_grad for them :
        self.register_buffer('mu', params['mu']) # mean of the growth functions (B,C,C)
        self.register_buffer('sigma', params['sigma']) # standard deviation of the growths functions (B,C,C)
        self.register_buffer('beta',params['beta']) # max of the kernel rings (B,C,C, # of rings)
        self.register_buffer('mu_k',params['mu_k'])# mean of the kernel gaussians (B,C,C, # of rings)
        self.register_buffer('sigma_k',params['sigma_k'])# standard deviation of the kernel gaussians (B,C,C, # of rings)
        self.register_buffer('weights',params['weights']) # raw weigths for the growth weighted sum (B,C,C)

        self.norm_weights()
        self.register_buffer('kernel',torch.zeros((self.k_size,self.k_size)))
        self.kernel = self.compute_kernel() # (B,C,C,h, w)


    def update_params(self, params, k_size_override = None):
        """
            Updates some or all parameters of the automaton. 
            Changes batch size to match the one of provided params (take mu as reference)
        """
        self.mu = params.get('mu',self.mu)
        self.sigma = params.get('sigma',self.sigma)
        self.beta = params.get('beta',self.beta)
        self.mu_k = params.get('mu_k',self.mu_k)
        self.sigma_k = params.get('sigma_k',self.sigma_k)
        self.weights = params.get('weights',self.weights)
        self.k_size = params.get('k_size',self.k_size) # kernel sizes (same for all)
        if(k_size_override is not None):
            self.k_size = k_size_override

        self.norm_weights()

        self.batch = self.mu.shape[0] # update batch size
        self.kernel = self.compute_kernel() # (B,C,C,h,w)

    
    def norm_weights(self):
        """
            Normalizes the relative weight sum of the growth functions
            (A_j(t+dt) = A_j(t) + dt G_{ij}w_ij), here we enforce sum_i w_ij = 1
        """
        # Normalizing the weights
        N = self.weights.sum(dim=1, keepdim = True) # (B,1,C)
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
            Sets the initial state of the automaton using fractal perlin noise.
            Max wavelength is k_size*1.5, chosen a bit randomly
        """
        self.state = perlin_fractal((self.batch,self.h,self.w),int(self.k_size*1.5),
                                    device=self.device,black_prop=0.25,num_channels=self.C,persistence=0.4) 
    
    def set_init_perlin(self,wavelength=None):
        """
            Sets initial state using one-wavelength perlin noise.
            Default wavelength is 2*K_size
        """
        if(not wavelength):
            wavelength = self.k_size
        self.state = perlin((self.batch,self.h,self.w),[wavelength]*2,
                            device=self.device,num_channels=self.C,black_prop=0.25)
        
    def kernel_slice(self, r):
        """
            Given a distance matrix r, computes the kernel of the automaton.
            In other words, compute the kernel 'cross-section' since we always assume
            rotationally symmetric kernel

            Args :
            r : (k_size,k_size), value of the radius for each pixel of the kernel
        """
        # Expand radius to match expected kernel shape
        r = r[None, None, None,None] #(1,1, 1, 1, k_size, k_size)
        r = r.expand(self.batch,self.C,self.C,self.mu_k.shape[3],-1,-1) #(B,C,C,#of rings,k_size,k_size)

        mu_k = self.mu_k[..., None, None] # (B,C,C,#of rings,1,1)
        sigma_k = self.sigma_k[..., None, None]# (B,C,C,#of rings,1,1)

        K = torch.exp(-((r-mu_k)/sigma_k)**2/2) #(B,C,C,#of rings,k_size,k_size)
        #print(K.shape)

        beta = self.beta[..., None, None] # (B,C,C,#of rings,1,1)
        K = torch.sum(beta*K, dim = 3) #

        
        return K #(B,C,C,k_size, k_size)
    
    def compute_kernel(self):
        """
            Computes the kernel given the current parameters.
        """
        xyrange = torch.arange(-1, 1+0.00001, 2/(self.k_size-1)).to(self.device)
        X,Y = torch.meshgrid(xyrange, xyrange,indexing='ij')
        r = torch.sqrt(X**2+Y**2)

        K = self.kernel_slice(r) #(B,C,C,k_size,k_size)

        # Normalize the kernel, s.t. integral(K) = 1
        summed = torch.sum(K, dim = (-1,-2), keepdim=True) #(B,C,C,1,1)

        # Avoid divisions by 0
        summed = torch.where(summed<1e-6,1,summed)
        K /= summed

        return K #(B,C,C,k_size,k_size)
    
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
        """
        # Shenanigans to make all the convolutions at once.
        kernel_eff = self.kernel.reshape([self.batch*self.C*self.C,1,self.k_size,self.k_size])#(B*C^2,1,k,k)

        U = self.state.reshape(1,self.batch*self.C,self.h,self.w) # (1,B*C,H,W)
        U = F.pad(U, [(self.k_size-1)//2]*4, mode = 'circular') # (1,B*C,H+pad,W+pad)
        
        U = F.conv2d(U, kernel_eff, groups=self.C*self.batch).squeeze(1) #(B*C^2,1,H,W) squeeze to (B*C^2,H,W)
        U = U.reshape(self.batch,self.C,self.C,self.h,self.w) # (B,C,C,H,W)

        assert (self.h,self.w) == (self.state.shape[2], self.state.shape[3])

        weights = self.weights[...,None, None] # (B,C,C,1,1)
        weights = weights.expand(-1,-1, -1, self.h,self.w) # (B,C,C,H,W)

        # Weight normalized growth :
        dx = (self.growth(U)*weights).sum(dim=1) #(B,C,H,W)

        # Apply growth and clamp
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
            Draws the RGB worldmap from state.
        """
        assert self.state.shape[0] == 1, "Batch size must be 1 to draw"
        toshow= self.state[0].permute((2,1,0)) # (W,H,C) for pygame

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