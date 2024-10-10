"""
    Script to generate individual videos of the automaton, given saved parameters.
"""


from torchenhanced.util import saveTensVideo
from modules import BatchLeniaMC
import os
from tqdm import tqdm
import torch
from modules.utils.main_utils import load_params

# param_dir = 'data/latest_rand/individual' # Directory containing the individual (unbatched) parameters
out_dir = 'test_videos' # Directory to save the videos
param_dir = 'data/demo_params/individual'

simulation_time = 1000 # Number of frames to simulate
size = 400,4500 # Size of the simulation
fps=120 # Framerate of the video
device='cuda:0' # Device on which to simulate


#### DO NOT MODIFY BELOW THIS LINE ####
out_dir = os.path.join('data/videos',out_dir)
os.makedirs(out_dir,exist_ok=True)

params = os.listdir(param_dir)
params_paths = [os.path.join(param_dir,v) for v in params]
param_names = [v.split('.')[0] for v in params]

auto = BatchLeniaMC((1,*size),0.1,device=device)

for i,p in tqdm(enumerate(params_paths),total=len(params_paths)):
    auto.update_params(load_params(p,make_batch=True,device=device))
    auto.set_init_perlin()

    v_tens = torch.zeros((1,simulation_time,3,*size),device=device)
    
    for t in range(simulation_time):
        v_tens[:,t] = auto.state
        auto.step()

    saveTensVideo(v_tens[0],out_dir,name=param_names[i],fps=fps,out_size=size[0],columns=1)