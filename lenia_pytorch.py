import numpy as np
import einops as eo
import matplotlib.pyplot as plt
import matplotlib.animation
import torch, torch.nn as nn
import time
import pickle
from torchenhanced.util import showTens

# defining the orbium
orbium = [[
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 4,13,10, 5, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 7, 4, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 2, 3, 1, 0, 0, 0, 0, 0, 0, 0, 1, 4, 1, 0, 0, 0],
[ 0, 1, 0, 1, 1, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 1, 0, 0],
[ 1, 0, 2, 3, 4, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0],
[ 0, 1, 3, 4, 6, 5, 4, 3, 4, 3, 0, 0, 0, 0, 0, 0, 0, 2, 4, 2, 0],
[ 0, 0, 3, 4, 6, 6, 3, 5, 6, 7, 8, 8, 6, 0, 0, 0, 0, 0, 3, 2, 0],
[ 0, 0, 2, 4, 6, 4, 4, 6, 7, 8,10,11,11,12,10, 7, 3, 3, 3, 3, 1],
[ 0, 0, 1, 3, 3, 1, 4, 6, 7, 9,11,12,13,13,12,10, 9, 7, 5, 3, 1],
[ 0, 0, 2, 3, 1, 0, 1, 4, 7, 9,11,13,13,13,12,10, 8, 7, 5, 3, 2],
[ 0, 0, 2, 2, 0, 0, 0, 0, 3, 8,10,13,13,12,11,10, 8, 6, 5, 3, 1],
[ 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 5,10,13,12,10, 9, 7, 6, 4, 3, 1],
[ 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 8,11, 9, 8, 6, 5, 4, 2, 1],
[ 0, 4, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 8, 7, 5, 4, 3, 2, 0],
[ 0, 0, 0, 5, 7, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 4, 3, 2, 0, 0],
[ 0, 0, 0, 0, 0, 7,11, 0, 0, 0, 0, 0, 1, 2, 3, 4, 3, 2, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 6, 7, 4, 2, 1, 3, 4, 3, 2, 1, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0],
]]
orbium = np.asarray(orbium)

# utils
def safe_divide(x, y, eps=1e-10):
    return x / (y + eps)

def gaussian_func(x, m, s, h=1):
    return np.exp(-safe_divide((x - m), s)**2 / 2) * h

def get_image(X, max_y=None, scale=1):
    n_dim = len(X.shape) - 1
    c = X.shape[0]
    y = X.shape[2]
    if max_y is not None:
        if y > max_y:
            X = X[:, :, -max_y:]
        elif y < max_y:
            X = np.pad(X, [(0,0), (0,0), (0,max_y-y)])
    # print(X.shape, X.dtype, X.max(), X.min())
    if c == 2:
        X = np.pad(X, [(0,1), (0,0), (0,0)], 'constant')
    elif c == 1:
        X = np.repeat(X, 3, axis=0)
    return np.clip(X / scale, 0, 1).transpose(2,1,0)  # [c x y] -> [y x c]

def display_video(A, title=None, step=1, cmap='viridis', video_path='../video/lenia.gif'):
    fig = plt.figure(figsize=(4,4), dpi=75, frameon=False)
    img = plt.imshow(get_image(A[0]), cmap=cmap, interpolation="nearest", vmin=0)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    if title is not None:
        title = fig.text(0.05, 0.95, title, bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, ha="left")
    plt.close()
    def animate(i):
        # title.set_text(str(i))
        img.set_data(get_image(A[i*step]))
    anim = matplotlib.animation.FuncAnimation(fig, animate, frames=A.shape[0]//step, interval=20)
    anim.save(video_path, writer=matplotlib.animation.PillowWriter(fps=30))

def load_pattern(pattern, array_sizes):
    array_mids = [size // 2 for size in array_sizes]
    array = np.zeros([1] + array_sizes, dtype=int)
    pattern = np.asarray(pattern)
    _, w, h = pattern.shape
    x1 = array_mids[0] - w//2;  x2 = x1 + w
    y1 = array_mids[1] - h//2;  y2 = y1 + h
    array[:, x1:x2, y1:y2] = pattern
    return array

def discretize(array, div=1, mult=True):
    if mult:
        return np.round(array * div).astype(int)
    else:
        return array.astype(int)

# create kernel and growth function from parameters
def get_rule(rule_params, kernel_params, growth_params, kernel_radius, device='cpu'):

    array_sizes = [2*kernel_radius+1, 2*kernel_radius+1]
    n_dim, space_scale, time_scale, array_div, kernel_div, growth_div, n_kernels = rule_params
    _space = tuple(range(-n_dim, 0))
    _111 = ' '.join(['1'] * int(n_dim))
    _xyz = ' '.join('x' + str(d + 1) for d in range(n_dim))
    array_mids = [size // 2 for size in array_sizes]

    kernel_params = np.asarray(kernel_params, dtype=float)  # [p]
    kernel_params = eo.repeat(kernel_params, 'p -> p k', k=n_kernels)  # [p k]

    growth_params = np.asarray(growth_params, dtype=float)  # [p]
    growth_params = eo.repeat(growth_params, 'p -> p k', k=n_kernels)  # [p k]

    kernel_func = gaussian_func
    growth_func = gaussian_func
    clip_func = np.clip

    # calculate distance from origin
    ranges = [slice(0 - mid, size - mid) for size, mid in zip(array_sizes, array_mids)]
    space = np.asarray(np.mgrid[ranges], dtype=float)  # X [d xyz]
    distance = np.linalg.norm(space, axis=0)  # D [xyz]

    # calculate kernel K
    distance_scaled = distance / space_scale  # [xyz]
    kernel_params_fix = eo.rearrange(kernel_params, f'... -> ... {_111}')  # [p k xyz]
    kernel = kernel_func(distance_scaled, *kernel_params_fix)  # [k xyz]
    kernel = discretize(kernel, kernel_div, mult=True)
    kernel_sum = np.sum(kernel, axis=_space, keepdims=True)  # [k 111]
    # print("kernel sum: ", kernel_sum.shape, kernel_sum)

    # calculate growth lookup table for g
    kernel_all = np.sum(kernel_sum, axis=_space)  # N_k [k]
    # print("kernel all: ", kernel_all.shape, kernel_all)
    max_volume = np.prod(np.asarray(array_sizes))
    percept_div = array_div * kernel_all  # [k]
    # print("percept div:", percept_div[0])
    percept_max = array_div * kernel_div * max_volume  # [1]
    # print(f"percept_max: {array_div} * {kernel_div} * {max_volume} = {percept_max}")
    percept_list = [ np.pad(np.linspace(0, 1, div), [0, percept_max - div]) for div in percept_div ]
    percept = np.asarray(percept_list)  # [k u]
    # print("percept: ", percept)
    growth_params_fix = eo.rearrange(growth_params, f'... -> ... 1')  # [p k u]
    growth_lookup = growth_func(percept, *growth_params_fix)  # [k u]


    growth_lookup = discretize(growth_lookup, growth_div, mult=True)[0]
    growth_support = [i for i in range(len(growth_lookup)) if growth_lookup[i]>0]
    if growth_support:
        arg_min, arg_max = min(growth_support), max(growth_support)
        g = [arg_min-1, arg_max+1]
    else:
        g = [0, 0]

    r = kernel_radius
    k = torch.tensor(np.asarray(kernel)).view(1, 1, 2 * r + 1, 2 * r + 1).to(device)

    # showTens(k.to(dtype=torch.float32))
    # conv = nn.Conv2d(1, 1, 2 * r + 1, bias=False, padding=r, padding_mode='circular')
    # for param in conv.parameters():
    #     param.requires_grad = False
    # conv.weight.data = k.clone().detach().float()



    return [k, g]

# one step of evolution
def step(array, conv, g, array_div, device='cpu'):
    # K*A
    with torch.no_grad():
        x = conv(array)

    # g(K*A)
    g_min, g_max = g
    x = ((x > g_min) & (x < g_max)).float()
    rate = x * 2 - 1

    # A + d/dt g(K*A)
    array_new = array + rate
    array_new = torch.clip(array_new, 0, array_div).float().to(device) # clip[A']

    return array_new



if __name__ == '__main__':
    # these are the parameters for discrete Lenia where the orbium occurs
    n_dim = 2
    space_scale = 13  # space resolution = radius of kernel
    time_scale = 1    # time resolution
    array_div  = 13   # number of levels in world values [0 ... P_a]
    kernel_div = 13   # number of levels in kernel values [0 ... P_k]
    growth_div = 1    # number of levels in growth values [0 ... P_g]
    n_kernels = 1

    kernel_radius = space_scale
    kernel_params = [0.5, 0.15, 1.0]
    growth_params = [0.15, 0.015, 1.0]

    rule_params = [n_dim, space_scale, time_scale, array_div, kernel_div, growth_div, n_kernels]
    device = 'cpu'



    # example of usecase:
    array_sizes=[100, 100]
    rand_size = 40 # creates a random 40 by 40 region, rest is zeros

    conv, g = get_rule(rule_params, kernel_params, growth_params, kernel_radius, device)
    # pattern = orbium
    pattern = np.random.randint(0, array_div+1, (1, rand_size, rand_size))
    init_array = load_pattern(pattern, array_sizes)
    init_array = torch.tensor(np.asarray(init_array)).view(1, 1, *array_sizes).to(device).float()


    time_steps = 100
    Y = torch.zeros(time_steps, 1, *array_sizes).float()
    Y[0, 0, :, :] = init_array[0]

    array = init_array

    for t in range(time_steps-1):
        array = step(array, conv, g, array_div, device)
        Y[t, 0, :, :] = array


    # to display the video uncomment this:
    # display_video(np.asarray(Y) / array_div, step=1, cmap='viridis', video_path=f'./{rand_size}_{array_sizes[0]}.gif')

