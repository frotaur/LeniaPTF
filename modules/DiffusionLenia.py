import torch,torch.nn,torch.nn.functional as F
from .Lenia import BatchLeniaMC


class DiffusionLenia(BatchLeniaMC):
    """
        Mass conserving Lenia-like Alife model, inspired from the discretization
        of the diffusion equation
    """

    def __init__(self, size, dt, num_channels=3, params=None, state_init=None, device='cpu'):
        """
            Args:
                size : tuple, (C,H,W) size of the automaton
                dt : float, time step size
                num_channels : int, number of channels
                params : dict, parameters of the automaton
                state_init : tensor, initial state of the automaton
                device : str, device to use
        """
        super(DiffusionLenia, self).__init__(size, dt, num_channels, params, state_init, use_fft=True, device=device)

        self._temp = 1
        self.Aff = self.compute_affinity()

    def step(self):
        """
            Steps the alife model by one time step
        """
        B,C,H,W = self.state.shape
        Aff = self.compute_affinity()
    
        Z = F.pad(Aff, (1,1,1,1), mode='circular') # (B,C,H+2,W+2) for the (3,3) kernel
        Z = F.unfold(Z, kernel_size=(3,3)).reshape(B,C,9,H,W) # (B,C*9,H,W)
        Z = Z.sum(dim=2) # (B,C,H,W) local affinity normalization

        state_portions = self.state/Z
        state_portions = F.pad(state_portions, (1,1,1,1), mode='circular') # (B,C,H+2,W+2) for the (3,3) kernel
        state_portions = F.unfold(state_portions, kernel_size=(3,3)).reshape(B,C,9,H,W) # (B,C*H*W,9)
        self.state = (Aff[:,:,None]*state_portions).sum(dim=2) # (B,C,H,W) result of the diffusion

    def compute_affinity(self):
        """
            Computes the affinity matrix of the model
        """
        Aff = self.get_fftconv(self.state) # (B,C,C,H,W) first step affinity, usual convolutions

        weights = self.weights[...,None, None] # (B,C,C,1,1)
        Aff = (self.growth(Aff)*weights).sum(dim=1) # (B,C,H,W) pre-exponential affinity
        Aff = torch.exp(self.temp*Aff) 

        return Aff

    @property
    def temp(self):
        return self._temp

    @temp.setter
    def temp(self, value):
        self._temp = value
    

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
        
        self._worldmap= torch.clip(toshow,min=0,max=1).cpu().numpy()   
