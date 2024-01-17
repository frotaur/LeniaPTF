import torch
import numpy as np, math
import perlin_numpy as pnp
from torchenhanced.util import saveTens

def perlin(size, wavelengths, black_prop=0.3,device='cpu'):
    """
        Returns a perlin noise image tensor (3,H,W).

        Args:
        size : (H,W) tuple or array, size of tensor
        wavelength : (lam_x,lam_y) tuple or array of ints, wavelengths (in pixels) in x and y of the noise
        black_prop : percentage of black (0) regions. Defaut is .5=50% of black.
        device : device for returned tensor

        returns :
        (3,H,W) torch tensor of perlin noise.
    """
    H,W = tuple(size)
    l_x, l_y = tuple(int(lam) for lam in wavelengths)
    
    # Extend image so that its integer wavelengths of noise
    W_new=int(W+(l_x-W%l_x))
    H_new=int(H+(l_y-H%l_y))

    n_x = int(W_new/l_x)
    n_y = int(H_new/l_y)

    print(f'using n : {(n_x,n_y)} with shape {(H_new,W_new)}, wavelength {l_x,l_y}')

    state_R = torch.tensor(pnp.generate_perlin_noise_2d((H_new,W_new), (n_y, n_x)),device = device, dtype = torch.float)
    state_G = torch.tensor(pnp.generate_perlin_noise_2d((H_new,W_new), (n_y, n_x)),device = device, dtype = torch.float)
    state_B = torch.tensor(pnp.generate_perlin_noise_2d((H_new,W_new), (n_y, n_x)),device = device, dtype = torch.float)
    state = torch.clamp((torch.stack((state_R, state_G, state_B))+(0.5-black_prop)*2)/(2*(1.-black_prop)),0,1)
    state = state[:, :H, :W]

    return state


def perlin_fractal(shape:tuple, max_wavelength:int, persistence=0.5,black_prop:float=0.3,device='cpu'):
    """
        Makes fractal noise with dominant wavelength equal to max_wavelength.
    """
    max_num = min(6,int(math.log2(max_wavelength)))

    return sum([persistence**(i+1)*perlin(shape,[int(2**(-i)*max_wavelength)]*2,black_prop=black_prop,device=device) for i in range(max_num)])

if __name__=='__main__':
    size = 400,400
    device = 'cpu'
    waves = np.array([40,40])

    tens = perlin_fractal(size,waves[0],black_prop=0.3,persistence=0.5)
    saveTens(tens[0:1],'.','jeff')
    # saveTens(0.75*perlin(size,freqs, device)+0.25*perlin(size,freqs'2,device),'.','jeff')


    
