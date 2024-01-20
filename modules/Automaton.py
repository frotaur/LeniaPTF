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

class ConwayGoL(Automaton):
    """
        Conway Game of life using Torch

        Parameters :
        size : 2-uple (W,H)
    """

    def __init__(self, size):
        super().__init__(size)
        # 0,1,2,3 of the first dimension are the N,W,S,E directions
        self.state = torch.randint(0,2,(1,self.h,self.w)) # (1, h, w)
        self.kernel = torch.tensor([[1,1,1], [1,0,1], [1,1,1]]) # (h=3, w=3)
        self.kernel = self.kernel[None, None] # (1,1,3,3)
        
        
                    
    def step(self):
        """
            Steps the automaton state by one iteration.

            Completely free to be changed
        """
        Sum = F.conv2d(self.state[None], self.kernel, padding = 'same')#(1,1,h,w)
        Sum = Sum.squeeze(0) #(1,H,W)
        #print('state avant', self.state)

        self.state = torch.where( self.state ==1, torch.where( torch.logical_and((2 <= Sum), ( Sum <= 3)),  self.state, torch.zeros_like(self.state)), torch.where(Sum == 3, torch.ones_like(self.state), self.state) )
        #print('state apres', self.state)

        toshow= self.state.permute((2,1,0)) #(W,H,1)
        toshow = toshow.expand(-1,-1,3) #(W,H,3)

        self._worldmap =np.zeros_like(self._worldmap) #(W,H,3)
        self._worldmap= toshow.numpy()

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

        self.register_buffer('kernel',torch.zeros((self.k_size,self.k_size)))
        self.kernel = self.compute_kernel() # (3,3,h, w)

        # Uncomment to show kernel
        #showTens(self.kernel.cpu().detach()*100)
        
        x = np.arange(0,1,0.01)

        # Uncomment to show growth functions
        # x = np.arange(0,1,0.01)
        # fig, axs = plt.subplots(3, 3)
        # RGB = ['R', 'G', 'B']
        # for i in range(3):
        #     for j in range(3):
        #         axs[i,j].plot(x, 2*np.exp(-((x-self.mu[i][j].item())/self.sigma[i][j].item())**2/2)-1)
        #         axs[i,j].set_title('Growth : ' + RGB[i] + '->' +RGB[j])
        # fig.show()


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

        K = torch.exp(-((r-mu_k)/2)**2/sigma_k) #(3,3,#of rings,k_size,k_size)
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

        return 2*torch.exp(-((u-mu)/(sigma))**2/2)-1 #(3,3,H,W)


    def step(self):
        """
            Steps the automaton state by one iteration.
        """
        # Shenanigans to make all the convolutions at once.
        kernel_eff = self.kernel.reshape([9,1,self.k_size,self.k_size])#(9,1,k,k)
        # kernel_eff = torch.zeros_like(kernel_eff)
        # kernel_eff[]

        U = F.pad(self.state[None], [(self.k_size-1)//2]*4, mode = 'circular')
        U = F.conv2d(U, kernel_eff, groups=3).squeeze(0) #(9,H,W)
        U = U.reshape(3,3,self.h,self.w)

        assert (self.h,self.w) == (self.state.shape[1], self.state.shape[2])
        # Maybe change to weighted sum with the weights being also parameters  (done)
        # factor = torch.eye(3)[...,None,None].to(self.device) #(3,3)
        
        # Normalizing the weights each time is wasteful; do it in the init.
        N = self.weights.sum(dim=0, keepdim = True)
        # print(N)
        weights = torch.where(N > 1.e-6, self.weights/N, 0)
        # print(weights)
        weights = weights [...,None, None]
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
    
    def batch_evolve_state(self,states,num_steps):
        """
            Evolves a batch of states for a given number of steps.
            Args :
            states : (B,3,H,W) tensor, batch of states
            num_steps : int, number of steps to evolve the states
        """
        B = states.shape[0]
        kernel_eff = self.kernel.reshape([9,1,self.k_size,self.k_size])[None].expand(B,-1,-1,-1,-1)#(B,9,1,k,k)

        U = F.pad(states, [(self.k_size-1)//2]*4, mode = 'circular') # (B,3,H+pad,W+pad)
        U = F.conv2d(U, kernel_eff, groups=3).squeeze(0) #(9,H,W)
        U = U.reshape(3,3,self.h,self.w)        

    def draw(self):
        """
            Draws the worldmap from state.
            Separate from step so that we can freeze time,
            but still 'paint' the state and get feedback.
        """
        
        toshow= self.state.permute((2,1,0)) #(W,H,3)
        W,H,_ = toshow.shape
        # toshow = toshow.expand(-1,-1,3) #(W,H,3)
        
        # toshow.reshape((H*W)) # (W*H*3)

        # cmap = plt.get_cmap("viridis")
        # toshow = cmap(toshow.cpu().numpy()) # (W*H,4)
        # toshow = toshow.reshape((W,H,4))[:,:,:3]
        
        # self._worldmap =np.zeros_like(self._worldmap) #(W,H,3)
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
    def __init__(self, size, dt, params=None, state_init = None, device='cpu' ):
        """
            Initializes automaton.  

            Args :
                size : (B,H,W) of ints, size of the automaton and number of batches
                dt : time-step used when computing the evolution of the automaton
                params : dict of tensors containing the parameters. If none, generates randomly
                    keys-values : 
                    'k_size' : odd int, size of kernel used for computations
                    'mu' : (B,3,3) tensor, mean of growth functions
                    'sigma' : (B,3,3) tensor, standard deviation of the growth functions
                    'beta' :  (B,3,3, # of rings) float, max of the kernel rings 
                    'mu_k' : (B,3,3, # of rings) [0,1.], location of the kernel rings
                    'sigma_k' : (B,3,3, # of rings) float, standard deviation of the kernel rings
                    'weights' : (B,3,3) float, weights for the growth weighted sum
                device : str, device 
        """
        super().__init__()
        self.to(device)

        self.batch= size[0]
        self.h, self.w  = size[1:]
        # 0,1,2,3 of the first dimension are the N,W,S,E directions
        if(params is None):
            params = gen_batch_params(self.batch,device)
        self.k_size = params['k_size'] # kernel sizes (same for all)

        self.register_buffer('state',torch.rand((self.batch,3,self.h,self.w)))

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
            Updates the parameters of the automaton. Changes batch size to match one of provided params.
        """
        self.mu = params['mu'] # mean of the growth functions (3,3)
        self.sigma = params['sigma'] # standard deviation of the growths functions (3,3)
        self.beta = params['beta']
        self.mu_k = params['mu_k']
        self.sigma_k = params['sigma_k']
        self.weights = params['weights']
        self.norm_weights()
        self.kernel = self.compute_kernel() # (B,3,3,h, w)

        self.batch = params['mu'].shape[0] # update batch size

    def norm_weights(self):
        # Normalizing the weights
        N = self.weights.sum(dim=0, keepdim = True)
        self.weights = torch.where(N > 1.e-6, self.weights/N, 0)

    def get_params(self):
        """
            Get the parameter dictionary which defines the automaton
        """
        params = dict(mu = self.mu, sigma = self.sigma, beta = self.beta,
                       mu_k = self.mu_k, sigma_k = self.sigma_k, weights = self.weights)
        
        return params

    def set_init_fractal(self):
        """
            Sets the initial state of the automaton using perlin noise
        """
        self.state = perlin_fractal((self.batch,self.h,self.w),int(self.k_size*1.5),
                                    device=self.device,black_prop=0.25,persistence=0.4) 
    
    def set_init_perlin(self,wavelength=None):
        if(not wavelength):
            wavelength = self.k_size
        self.state = perlin((self.batch,self.h,self.w),[wavelength]*2,
                            device=self.device,black_prop=0.25)

    def kernel_slice(self, r): # r : (k_size,k_size)
        """
            Given a distance matrix r, computes the kernel of the automaton.

            Args :
            r : (k_size,k_size), value of the radius for each location around the center of the kernel
        """

        r = r[None, None, None,None] #(1,1, 1, 1, k_size, k_size)
        r = r.expand(self.batch,3,3,self.mu_k[0][0].size()[0],-1,-1) #(B,3,3,#of rings,k_size,k_size)

        mu_k = self.mu_k[..., None, None] # (B,3,3,#of rings,1,1)
        sigma_k = self.sigma_k[..., None, None]# (B,3,3,#of rings,1,1)

        K = torch.exp(-((r-mu_k)/2)**2/sigma_k) #(B,3,3,#of rings,k_size,k_size)
        #print(K.shape)

        beta = self.beta[..., None, None] # (B,3,3,#of rings,1,1)

        K = torch.sum(beta*K, dim = 2)

        
        return K #(B,3,3,k_size, k_size)
    
    def compute_kernel(self):
        """
            Computes the kernel given the parameters.
        """
        xyrange = torch.arange(-1, 1+0.00001, 2/(self.k_size-1)).to(self.device)
        X,Y = torch.meshgrid(xyrange, xyrange,indexing='ij')
        r = torch.sqrt(X**2+Y**2)

        K = self.kernel_slice(r) #(B,3,3,k_size,k_size)

        # Normalize the kernel
        summed = torch.sum(K, dim = (-1,-2), keepdim=True) #(B,3,3,1,1)

        # Avoid divisions by 0
        summed = torch.where(summed<1e-6,1,summed)
        K /= summed

        return K #(B,3,3,k,k)
    
    def growth(self, u): # u:(B,3,3,H,W)
        """
            Computes the growth of the automaton given the concentration u.

            Args :
            u : (B,3,3,H,W) tensor of concentrations.
        """

        # Possibly in the future add other growth function using bump instead of guassian
        mu = self.mu[..., None, None] # (B,3,3,1,1)
        sigma = self.sigma[...,None,None] # (B,3,3,1,1)
        mu = mu.expand(-1,-1,-1, self.h, self.w) # (B,3,3,H,W)
        sigma = sigma.expand(-1,-1,-1, self.h, self.w) # (B,3,3,H,W)

        return 2*torch.exp(-((u-mu)/(sigma))**2/2)-1 #(B,3,3,H,W)


    def step(self):
        """
            Steps the automaton state by one iteration.
        """
        # Shenanigans to make all the convolutions at once.
        kernel_eff = self.kernel.reshape([self.batch*9,1,self.k_size,self.k_size])#(B*9,1,k,k)

        U = self.state.reshape(1,self.batch*3,self.h,self.w) # (1,B*3,H,W)
        U = F.pad(U, [(self.k_size-1)//2]*4, mode = 'circular') # (1,B*3,H+pad,W+pad)
        
        U = F.conv2d(U, kernel_eff, groups=3*self.batch).squeeze(0) #(1,B*9,H,W) squeeze to (B*9,H,W)
        U = U.reshape(self.batch,3,3,self.h,self.w) # (B,3,3,H,W)

        # assert (self.h,self.w) == (self.state.shape[2], self.state.shape[3])
        
        weights = self.weights [...,None, None] # (B,3,3,1,1)
        weights = weights.expand(-1,-1, -1, self.h,self.w) # (B,3,3,H,W)

        dx = (self.growth(U)*weights).sum(dim=1) #(B,3,H,W)

        self.state = torch.clamp(self.state + self.dt*dx, 0, 1)     
        
    def mass(self):
        """
            Computes average 'mass' of the automaton for each channel

            returns :
            mass : (B,3) tensor, mass of each channel
        """

        return self.state.mean(dim=(-1,-2)) # (B,3) mean mass for each color
