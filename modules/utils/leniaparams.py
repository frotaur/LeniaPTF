import torch
import math, os
from ..utils import params_to_words

class LeniaParams():
    """
        Class handling parameters of the Lenia automaton.
    """

    def __init__(self, param_dict=None, from_file=None, k_size=None, batch_size=None, device='cpu'):
        """
            Args:
                from_file : str, path to file containing parameters. Prority over param_dict
                param_dict : dict, dictionary of parameters
                k_size : int, size of the kernel. Used if both param_dict and from_file are None
                device : str, device to use
        """
        self.device = device
        if(param_dict is None and from_file is None):
            # Not great, but ok for now
            assert k_size is not None and batch_size is not None, 'k_size and batch_size must be provided if no parameters are given'
            self.batch_size = batch_size
            self.k_size = k_size
            param_dict = self.default_gen().param_dict # dis very ugly but not sure how to do it better

        if(from_file is not None):
            param_dict = torch.load(from_file, map_location=device)

        assert param_dict.keys() == {'k_size','mu','sigma','beta','mu_k','sigma_k','weights'}, f'Invalid parameter dictionary, got keys : {param_dict.keys()}'
        
        self.k_size = param_dict['k_size']
        self.mu = param_dict['mu'].to(device)
        self.sigma = param_dict['sigma'].to(device)
        self.beta = param_dict['beta'].to(device)
        self.mu_k = param_dict['mu_k'].to(device)
        self.sigma_k = param_dict['sigma_k'].to(device)
        self.weights = param_dict['weights'].to(device)

        self.batch_size = self.mu.shape[0]


        self._sanitize()
    
    @property
    def param_dict(self):
        """
            Returns the parameters as a dictionary.
        """
        return {
            'k_size' : self.k_size,
            'mu' : self.mu,
            'sigma' : self.sigma,
            'beta' : self.beta,
            'mu_k' : self.mu_k,
            'sigma_k' : self.sigma_k,
            'weights' : self.weights
        }
    
    def save_indiv(self, folder, params, batch_name=False, annotation=None):
        """
            Saves parameter both in batch and individually.

            Args:
            folder : path to folder where to save params individually
            params : dictionary of parameters
            batch_name : if True, names indiv parameters with batch name + annotation
            annotation : list of same length as batch_size, an annotation of the parameters
        """
        name = params_to_words(self.param_dict)
        batch_size = self.batch_size

        params_list = [params[i:i+1] for i in range(batch_size)] # to keep the batch size

        if(annotation is None):
            annotation = [f'{j:02d}' for j in range(len(params_list))]
        else:
            assert len(annotation)==len(params_list), f'Annotation (len={len(annotation)}) must \
            have same length as batch_size (len={len(params_list)})'

        for j in range(len(params_list)):
            if(not batch_name):
                indiv_name = params_to_words(params_list[j])
                fullname = indiv_name
            else:
                fullname = name+f'_{annotation[j]}'+'.pt'
            
            torch.save(params_list[j],os.path.join(fullname))

    def save(self, path):
        """
            Saves the (batched) parameters to a file.
        """
        torch.save(self.param_dict, path)
    
    def load(self, path):
        """
            Loads the parameters from a file.
        """
        self.__init__(from_file=path)

    def _sanitize(self):
        """
            Sanitizes the parameters by clamping them to valid values.
        """
        self.mu = torch.clamp(self.mu,0,2)
        self.sigma = torch.clamp(self.sigma,0,None)
        self.beta = torch.clamp(self.beta,0,1)
        self.mu_k = torch.clamp(self.mu_k,0,2)
        self.sigma_k = torch.clamp(self.sigma_k,0,None)
        self.weights = torch.clamp(self.weights,0,None)

    def __mul__(self, scalar: float)-> 'LeniaParams':
        """
            Multiplies all parameters by a scalar.
        """
        new_params = {}
        for key in self.param_dict.keys():
            if(key=='k_size'):
                new_params[key] = self.param_dict[key]
            new_params[key] = self.param_dict[key]*scalar
        
        return LeniaParams(param_dict=new_params,device=self.device)

    
    def __getitem__(self, idx):
        """
            Works as a dictionary, indexing the parameters with strings, or
            as advanced indexing on the batch dimension, like in pytorch
        """
        if(isinstance(idx, str)):
            return self.param_dict[idx]
        else:
            params ={}
            for k,v in self.param_dict.items():
                if(k=='k_size'):
                    params[k] = v
                else:
                    params[k] = v[idx]
        
            return LeniaParams(params,device=self.device)

    def __add__(self, other : 'LeniaParams'):
        """
            Adds two sets of parameters together.
        """
        assert self.batch_size == other.batch_size, 'Batch sizes do not match'
        assert self.k_size == other.k_size, 'Kernel sizes do not match'

        new_params = {}
        for key in self.param_dict.keys():
            if(key=='k_size'):
                new_params[key] = self.param_dict[key]
            new_params[key] = self.param_dict[key] + other.param_dict[key]
        
        return LeniaParams(param_dict=new_params,device=self.device)


    def cat(self, other : 'LeniaParams'):
        """
            Concatenates two sets of parameters together.
        """
        assert self.k_size == other.k_size, 'Kernel sizes do not match'
        assert self.device == other.device, f'Devices do not match, got {self.device} and {other.device}'
        new_params = {}
        for key in self.param_dict.keys():
            if(key=='k_size'):
                new_params[key] = self.param_dict[key]
            else :
                new_params[key] = torch.cat([self.param_dict[key], other.param_dict[key]], dim=0)
        
        return LeniaParams(param_dict=new_params,device=self.device)


    def mutate(self, magnitude = 0.02, rate = 0.1, frozen_keys = []):
        """
            Mutates the parameters by a small amount.

            Args:
                magnitude : float, magnitude of the mutation
                rate : float, will change a parameter with this rate
                frozen_keys : list of str, keys to not mutate
        """
        keys = list(self.param_dict.keys())

        new_params = {}
        for key in keys:
            if(key not in frozen_keys and key!='k_size'):
                tentative = self.param_dict[key]*(1 + magnitude*torch.randn_like(self.param_dict[key]))
                
                new_params[key] = torch.where(torch.rand_like(tentative)<rate, tentative, self.param_dict[key])
            else:
                new_params[key] = self.param_dict[key]
                
        
        return LeniaParams(param_dict=new_params, device=self.device)
    
    def default_gen(self,num_channels = 3, k_size=None):
        """
            Empirical parameter generations which works ok with lenia_ptf

            Args:
                batch_size : number of parameters to generate
                device : device on which to generate the parameters
            
            Returns:
                dict of batched parameters
        """
        mu = 0.7*torch.rand((self.batch_size,num_channels,num_channels), device=self.device) 
        sigma = mu/(math.sqrt(2*math.log(2)))*0.8*torch.rand((self.batch_size,num_channels,num_channels), device=self.device)+1e-4
        
        params = {
                'k_size' : k_size if k_size is not None else self.k_size, 
                'mu':  mu ,
                'sigma' : sigma,
                'beta' : torch.rand((self.batch_size,num_channels,num_channels,3), device=self.device), 
                'mu_k' : torch.clamp(0.5+0.2*torch.randn((self.batch_size,num_channels,num_channels,3), device=self.device),min=0.,max=1.2), 
                'sigma_k' : 0.05*(1+torch.clamp(0.3*torch.randn((self.batch_size,num_channels,num_channels,3), device=self.device),min=-0.9)+1e-4),
                'weights' : torch.rand(self.batch_size,num_channels,num_channels,device=self.device)*(1-0.8*torch.diag(torch.ones(num_channels,device=self.device)))
            }
        
        return LeniaParams(params)
