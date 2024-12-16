import torch,torch.nn,torch.nn.functional as F
import numpy as np
from torchenhanced import DevModule
from .utils.noise_gen import perlin,perlin_fractal
from .utils.leniaparams import LeniaParams
from showtens import show_image


class LightLenia(DevModule):
    def __init__(self, size, dt, num_channels=3, params=None, light_params=None, state_init = None, device='cpu' ):
        """
            Initializes automaton.  

            Args :
                size : (B,H,W) of ints, size of the automaton and number of batches
                dt : time-step used when computing the evolution of the automaton
                num_channels : int, number of channels (C) in the automaton
                params : LeniaParams class, or dict of parameters containing the following
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
            self.params = LeniaParams(batch_size=self.batch, k_size=26, channels=self.C-1, device=device)
        elif(isinstance(params,dict)):
            self.params = LeniaParams(param_dict=params, device=device)
        else:
            self.params = params
        
        if(light_params is None):
            self.light_params = {}

        self.k_size = self.params['k_size'] # kernel sizes (same for all) ODD for conv2d, even for fft
        self.light_k_size = self.k_size*5

        self.register_buffer('matter',torch.rand((self.batch,self.C-1,self.h,self.w)))
        self.register_buffer('light',torch.rand((self.batch,1,self.h,self.w)))

        if(state_init is None):
            self.set_init_fractal() # Fractal perlin init
        else:
            self.matter = state_init.to(self.device) # Specific init

        self.dt = dt

        # Buffer for all parameters since we do not require_grad for them :
        self.register_buffer('mu', self.params['mu']) # mean of the growth functions (B,C,C)
        self.register_buffer('sigma', self.params['sigma']) # standard deviation of the growths functions (B,C,C)
        self.register_buffer('beta',self.params['beta']) # max of the kernel rings (B,C,C, # of rings)
        self.register_buffer('mu_k',self.params['mu_k'])# mean of the kernel gaussians (B,C,C, # of rings)
        self.register_buffer('sigma_k',self.params['sigma_k'])# standard deviation of the kernel gaussians (B,C,C, # of rings)
        self.register_buffer('weights',self.params['weights']) # raw weigths for the growth weighted sum (B,C,C)
        self.register_buffer('kernel',torch.zeros((self.k_size,self.k_size)))

        # Light field params
        self.register_buffer('light_mu', torch.rand((self.batch,1,1)))
        self.register_buffer('light_sigma', torch.rand((self.batch,1,1)))
        self.register_buffer('light_mu_k', torch.rand((self.batch,1,1,1)))
        self.register_buffer('light_sigma_k', torch.rand((self.batch,1,1,1)))
        self.register_buffer('light_kernel', torch.zeros((self.light_k_size,self.light_k_size)))

        # Light-matter interaction
        self.register_buffer('light_to_matter', 0.1*torch.rand((self.batch,1)))
        self.register_buffer('matter_to_light', 0.1*torch.rand((self.batch,1)))


        self.update_params(self.params,self.light_params)

    def update_params(self, matter_params, light_params, k_size_override = None):
        """
            Updates some or all parameters of the automaton. 
            Changes batch size to match the one of provided params (take mu as reference)

            Args:
                params : LeniaParams or dict, prefer the former
        """
        if(isinstance(matter_params,LeniaParams)):
            matter_params = matter_params.param_dict

        # Lenia matter_params
        self.mu = matter_params.get('mu',self.mu)
        self.sigma = matter_params.get('sigma',self.sigma)
        self.beta = matter_params.get('beta',self.beta)
        self.mu_k = matter_params.get('mu_k',self.mu_k)
        self.sigma_k = matter_params.get('sigma_k',self.sigma_k)
        self.weights = matter_params.get('weights',self.weights)
        self.k_size = matter_params.get('k_size',self.k_size) # kernel sizes (same for all)
        
        if(k_size_override is not None):
            self.k_size = k_size_override
        if(self.k_size%2==0):
                self.k_size += 1
                print(f'Using fft, increased even kernel size to {self.k_size}')
        # light params
        self.light_mu = light_params.get('liht_mu',self.light_mu)
        self.light_sigma = light_params.get('light_sigma',self.light_sigma)
        self.light_mu_k = light_params.get('light_mu_k',self.light_mu_k)
        self.light_sigma_k = light_params.get('light_sigma_k',self.light_sigma_k)

        # Light-matter interaction
        self.light_to_matter = light_params.get('light_to_matter',self.light_to_matter)	
        self.matter_to_light = light_params.get('matter_to_light',self.matter_to_light)
    

        self.params = LeniaParams(param_dict=matter_params, device=self.device)

        self.norm_weights()

        self.batch = self.mu.shape[0] # update batch size
        self.kernel = self.compute_kernel() # (B,C,C,k_size,k_size)

        self.light_kernel = self.compute_light_kernel() # (B,1,1,k_size*5,k_size*5)

        self.fft_kernel = self.kernel_to_fft(self.kernel) # (B,C,C,h,w)
        self.fft_light_kernel = self.kernel_to_fft(self.light_kernel) # (B,1,1,h,w)
    
    def norm_weights(self):
        """
            Normalizes the relative weight sum of the growth functions
            (A_j(t+dt) = A_j(t) + dt G_{ij}w_ij), here we enforce sum_i w_ij = 1
        """
        # Normalizing the weights
        N = self.weights.sum(dim=1, keepdim = True) # (B,1,C)
        self.weights = torch.where(N > 1.e-6, self.weights/N, 0)

    def get_params(self) -> LeniaParams:
        """
            Get the LeniaParams which defines the automaton
        """
        return self.params

    def set_init_fractal(self):
        """
            Sets the initial state of the automaton using fractal perlin noise.
            Max wavelength is k_size*1.5, chosen a bit randomly
        """
        fullstate = perlin_fractal((self.batch,self.h,self.w),int(self.k_size*1.5),
                                    device=self.device,black_prop=0.25,num_channels=self.C,persistence=0.4) 
        self.matter = fullstate[:,:2]
        self.light = fullstate[:,2:]

    def set_init_perlin(self,wavelength=None):
        """
            Sets initial state using one-wavelength perlin noise.
            Default wavelength is 2*K_size
        """
        if(not wavelength):
            wavelength = self.k_size
        fullstate = perlin((self.batch,self.h,self.w),[wavelength]*2,
                            device=self.device,num_channels=self.C,black_prop=0.25)
        self.matter = fullstate[:,:2]
        self.light = fullstate[:,2:]
        
    def kernel_slice(self, r, mu_k=None, sigma_k=None, beta=None):
        """
            Given a distance matrix r, computes the kernel of the automaton.
            In other words, compute the kernel 'cross-section' since we always assume
            rotationally symmetric kernel

            Args :
            r : (k_size,k_size), value of the radius for each pixel of the kernel
        """
        if(mu_k is None):
            mu_k = self.mu_k
        if(sigma_k is None):
            sigma_k = self.sigma_k
        if(beta is None):
            beta = self.beta
        
        C = mu_k.shape[1]
        # Expand radius to match expected kernel shape
        r = r[None, None, None,None] #(1,1, 1, 1, k_size, k_size)
        r = r.expand(self.batch,C,C,self.mu_k.shape[3],-1,-1) #(B,C,C,#of rings,k_size,k_size)

        mu_k = mu_k[..., None, None] # (B,C,C,#of rings,1,1)
        sigma_k = sigma_k[..., None, None]# (B,C,C,#of rings,1,1)

        K = torch.exp(-((r-mu_k)/sigma_k)**2/2) #(B,C,C,#of rings,k_size,k_size)
        #print(K.shape)

        beta = beta[..., None, None] # (B,C,C,#of rings,1,1)
        K = torch.sum(beta*K, dim = 3) #

        
        return K #(B,C,C,k_size, k_size)
    
    def compute_kernel(self):
        """
            Computes the kernel given the current parameters.
        """
        xyrange = torch.linspace(-1, 1, self.k_size).to(self.device)

        X,Y = torch.meshgrid(xyrange, xyrange,indexing='xy') # (k_size,k_size),  axis directions is x increasing to the right, y increasing to the bottom
        r = torch.sqrt(X**2+Y**2)

        K = self.kernel_slice(r) #(B,C,C,k_size,k_size)

        # Normalize the kernel, s.t. integral(K) = 1
        summed = torch.sum(K, dim = (-1,-2), keepdim=True) #(B,C,C,1,1)

        # Avoid divisions by 0
        summed = torch.where(summed<1e-6,1,summed)
        K /= summed
        Kshow = torch.cat([K, torch.zeros_like(K)[:,:,0:1]], dim=2)
        # show_image(Kshow,rescale=True)

        return K #(B,C,C,k_size,k_size)
    
    def compute_light_kernel(self):
        """
            Computes the kernel given the current parameters.
        """
        xyrange = torch.linspace(-1, 1, self.light_k_size).to(self.device)

        X,Y = torch.meshgrid(xyrange, xyrange,indexing='xy') # (k_size,k_size),  axis directions is x increasing to the right, y increasing to the bottom
        r = torch.sqrt(X**2+Y**2)

        K = self.kernel_slice(r, mu_k=self.light_mu_k, sigma_k=self.light_sigma_k, beta=torch.ones_like(self.light_mu_k)) #(B,C,C,k_size,k_size)

        # Normalize the kernel, s.t. integral(K) = 1
        summed = torch.sum(K, dim = (-1,-2), keepdim=True) #(B,C,C,1,1)

        # Avoid divisions by 0
        summed = torch.where(summed<1e-6,1,summed)
        K /= summed

        # show_image(K,rescale=True)
        
        return K #(B,C,C,k_size,k_size)

    def kernel_to_fft(self, K):
        # Pad kernel to match image size
        k_size = K.shape[-1]
        # Pad kernel to match image size
        # For some reason, pad is left-right, top-bottom, (so W,H)
        K = F.pad(K, [0,(self.w-k_size)] + [0,(self.h-k_size)]) # (B,C,C,h,w)
        print('Padded kenel, shape : ', K.shape)
        # show_image(K,rescale=True)
        # Center the kernel on the top left corner for fft
        K = K.roll((-(k_size//2),-(k_size//2)),dims=(-1,-2)) # (B,C,C,h,w)
        # show_image(K,rescale=True)
        K = torch.fft.fft2(K) # (B,C,C,h,w)
        # show_image(torch.cat([torch.abs(K),torch.angle(K)],dim=0),rescale=True)

        return K #(B,C,C,h,w)

    def growth(self, u, mu, sigma): # u:(B,C,C,H,W)
        """
            Computes the growth of the automaton given the concentration u.

            Args :
            u : (B,C,C,H,W) tensor of concentrations.
        """

        # Possibly in the future add other growth function using bump instead of guassian
        mu = mu[..., None, None] # (B,C,C,1,1)
        sigma = sigma[...,None,None] # (B,C,C,1,1)
        mu = mu.expand(-1,-1,-1, self.h, self.w) # (B,C,C,H,W)
        sigma = sigma.expand(-1,-1,-1, self.h, self.w) # (B,C,C,H,W)

        return 2*torch.exp(-((u-mu)**2/(sigma)**2)/2)-1 #(B,C,C,H,W)


    def step(self):
        """
            Steps the automaton state by one iteration.
        """
        matter_conv = self.get_fftconv(self.matter, self.fft_kernel)
        light_conv = self.get_fftconv(self.light, self.fft_light_kernel)

        assert (self.h,self.w) == (self.matter.shape[2], self.matter.shape[3])

        weights = self.weights[...,None, None] # (B,C,C,1,1)
        weights = weights.expand(-1,-1, -1, self.h,self.w) # (B,C,C,H,W)

        # Weight normalized growth :
        dmatter = (self.growth(matter_conv+self.light_to_matter*self.light,mu=self.mu,sigma=self.sigma)*weights).sum(dim=1) #(B,C,H,W)
        dlight = (self.growth(light_conv+self.matter_to_light*self.matter.sum(dim=1,keepdim=True),mu=self.light_mu,sigma=self.light_sigma)).sum(dim=1) #(B,C,H,W)
        # Apply growth and clamp
        self.matter = torch.clamp(self.matter + self.dt*dmatter, 0, 1) # (B,C,H,W)
        self.light = torch.clamp(self.light + self.dt*dlight, 0, 1) # (B,C,H,W)
    
    def get_fftconv(self, state, fft_kernel):
        """
            Compute convolution using fft
        """
        state = torch.fft.fft2(state) # (B,C,H,W) fourier transform
        state = state[:,:,None] # (B,1,C,H,W)
        state = state*fft_kernel # (B,C,C,H,W), convoluted
        state = torch.fft.ifft2(state) # (B,C,C,H,W), back to spatial domain

        return torch.real(state)

    def mass(self):
        """
            Computes average 'mass' of the automaton for each channel

            returns :
            mass : (B,C) tensor, mass of each channel
        """

        return self.matter.mean(dim=(-1,-2)) # (B,C) mean mass for each color

    def draw(self):
        """
            Draws the RGB worldmap from state.
        """
        toshow = torch.cat([self.matter, self.light], dim=1)[0]
        
        self._worldmap= toshow.permute((2,1,0)).cpu().numpy()   

    @property
    def worldmap(self):
        return (255*self._worldmap).astype(dtype=np.uint8)

