import torch
import numpy as np, math
import perlin_numpy as pnp
from pyperlin import FractalPerlin2D

def perlin(shape:tuple, wavelengths:tuple, black_prop:float=0.3,device='cpu'):
    """
        Makes fractal noise with dominant wavelength equal to max_wavelength.

        Args:
        shape : (B,H,W) tuple or array, size of tensor
        wavelength : int, wavelength of the noise in pixels
        black_prop : percentage of black (0) regions. Defaut is .5=50% of black.
        device : device for returned tensor

        Returns :
        (B,3,H,W) torch tensor of perlin noise.
    """    
    B,H,W = tuple(shape)
    lams = tuple(int(wave) for wave in wavelengths)
    # Extend image so that its integer wavelengths of noise
    W_new=int(W+(lams[0]-W%lams[0]))
    H_new=int(H+(lams[1]-H%lams[1]))
    frequency = [H_new//lams[0],W_new//lams[1]]
    gen = torch.Generator(device=device) # for GPU acceleration
    gen.seed()
    # Strange 1/0.7053 factor to get images noise in range (-1,1), quirk of implementation I think...
    fp = FractalPerlin2D((B*3,H_new,W_new), [frequency], [1/0.7053], generator=gen)()[:,:H,:W].reshape(B,3,H,W) # (B*3,H,W) noise)

    return torch.clamp((fp+(0.5-black_prop)*2)/(2*(1.-black_prop)),0,1)

def perlin_fractal(shape:tuple, max_wavelength:int, persistence=0.5,black_prop:float=0.3,device='cpu'):
    """
        Makes fractal noise with dominant wavelength equal to max_wavelength.
    """
    max_num = min(6,int(math.log2(max_wavelength)))
    normalization = float(sum([persistence**(i+1) for i in range(max_num)]))
    return 1./normalization*sum([persistence**(i+1)*perlin(shape,[int(2**(-i)*max_wavelength)]*2,black_prop=black_prop,device=device) for i in range(max_num)])

if __name__=='__main__':
    from time import time
    from torchenhanced.util import saveTensImage
    size = 400,400
    device = 'cpu'
    waves = np.array([60,60])

    t0 = time()
    tens = perlin_fractal((1,*size),waves[0],black_prop=0.5,device='cpu')
    saveTensImage(tens.cpu(),'.','fast')
    print('perlin fast : ', time()-t0)

    t0 = time()
    for i in range(1):
        tens = perlin_fractal(size,waves[0],black_prop=0.5)
    saveTensImage(tens,'.','slow')
    print('perlin : ', time()-t0)


def perlin_old(size, wavelengths, black_prop=0.3,device='cpu'):
    """
        Returns a perlin noise image tensor (3,H,W). DEPRECATE

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

    # print(f'using n : {(n_x,n_y)} with shape {(H_new,W_new)}, wavelength {l_x,l_y}')

    state_R = torch.tensor(pnp.generate_perlin_noise_2d((H_new,W_new), (n_y, n_x)),device = device, dtype = torch.float)
    state_G = torch.tensor(pnp.generate_perlin_noise_2d((H_new,W_new), (n_y, n_x)),device = device, dtype = torch.float)
    state_B = torch.tensor(pnp.generate_perlin_noise_2d((H_new,W_new), (n_y, n_x)),device = device, dtype = torch.float)
    fp =torch.stack((state_R, state_G, state_B))
    print('fp max an min : ', torch.max(fp), torch.min(fp))
    state = torch.clamp((torch.stack((state_R, state_G, state_B))+(0.5-black_prop)*2)/(2*(1.-black_prop)),0,1)
    state = state[:, :H, :W]

    return state


def perlin_fractal_old(shape:tuple, max_wavelength:int, persistence=0.5,black_prop:float=0.3,device='cpu'):
    """
        Makes fractal noise with dominant wavelength equal to max_wavelength.
        DEPRECATED
    """
    max_num = min(6,int(math.log2(max_wavelength)))

    return sum([persistence**(i+1)*perlin_old(shape,[int(2**(-i)*max_wavelength)]*2,black_prop=black_prop,device=device) for i in range(max_num)])

