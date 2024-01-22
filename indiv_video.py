from torchenhanced.util import saveTensVideo
from modules import LeniaMC
import os
from tqdm import tqdm
import torch
from modules.utils.main_utils import load_params

param_dir = 'data/latest/individual'
out_dir = 'rand_int_torank'


simulation_time = 1800
size = 500,500
fps=120
device='cuda:0'



out_dir = os.path.join('data/videos',out_dir)
os.makedirs(out_dir,exist_ok=True)

params = os.listdir(param_dir)
params_paths = [os.path.join(param_dir,v) for v in params]
param_names = [v.split('.')[0] for v in params]

auto = LeniaMC(size,0.1,device=device)

for i,p in tqdm(enumerate(params_paths),total=len(params_paths)):
    auto.update_params(load_params(p,device=device))
    auto.set_init_perlin()

    v_tens = torch.zeros((simulation_time,3,*size),device=device)
    
    for t in range(simulation_time):
        v_tens[t] = auto.state
        auto.step()

    saveTensVideo(v_tens,out_dir,name=param_names[i],fps=fps,out_size=size[0],columns=1)