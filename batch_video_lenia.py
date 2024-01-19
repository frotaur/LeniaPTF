from modules.utils.main_utils import gen_batch_params,load_params
from modules.Automaton import BatchLeniaMC
import torch
from torchenhanced.util import saveTensVideo

simulation_time = 640
size = 150,150
output_file = 'testing'
fps=60
param = load_params('data/paper_search/batch/bf1f8507.pk', device='cuda:0') # Parameter file to load

batch_size = param['mu'].shape[0]
print('BATCH SIZE IS : ', batch_size)
auto = BatchLeniaMC((batch_size,*size),0.1,param,device='cuda:0')
auto.set_init_perlin()

T, C, H, W = simulation_time,3, size[0],size[1]

tensor_out = torch.zeros((batch_size,T,C,H,W),dtype=torch.float32,device='cuda:0')

for i in range(simulation_time):
    auto.step()
    tensor_out[:,i,:,:,:] = auto.state

print(tensor_out.shape)
saveTensVideo(tensor_out,folderpath='.',name='output_file',fps=fps,out_size=800,columns=5)