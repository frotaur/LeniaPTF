from modules.utils.main_utils import gen_batch_params,load_params
from modules.Automaton import BatchLeniaMC
import torch
from torchenhanced.util import saveTensVideo, showTens, gridify
from torchvision.utils import make_grid
import cv2,numpy as np
from tqdm import tqdm
from time import time
import torchvision.transforms as transf

@torch.no_grad()
def save_batch_video(batch_param_location,name,save_name,bunching=15, columns=5, out_size=1200):
    """
        Save video of a batch of parameters

        Args:
            batch_param_location : location of the batch of parameters
            name : name of the video
            bunching : number of frames to compute at once (default 100)
    """
    simulation_time = 1000
    size = 400,400
    fps=120
    save_fold = os.path.join('data/videos',save_name)
    os.makedirs(save_fold,exist_ok=True)
    output_file = os.path.join(save_fold,f"{name}.mp4")

    param = load_params(batch_param_location, device='cuda:0') # Parameter file to load

    batch_size = param['mu'].shape[0]

    auto = BatchLeniaMC((batch_size,*size),0.1,param,device='cuda:0')
    auto.set_init_perlin()
    B,C,H,W = auto.state.shape
    _,gH,gW = gridify(auto.state,out_size=out_size,columns=columns).shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (gW, gH))

    step_times =0
    transmute_times = 0
    record_times = 0
    print('gridify shape', gridify(auto.state,out_size=out_size,columns=columns).shape)



    for i in tqdm(range(simulation_time//bunching)):
        buffer_tens = torch.zeros((batch_size,bunching,C,H,W),device='cuda:0')
        for i in range(bunching):
            t0 = time()
            auto.step()
            step_times+=time()-t0
            buffer_tens[:,i] = 255*auto.state
            transmute_times+=time()-t0

        t0 = time()
        # showTens(gridify(buffer_tens/255,out_size=out_size,columns=8)[0].cpu())
        grid_tens = gridify(buffer_tens,out_size=out_size,columns=columns).permute(0,2,3,1).cpu().numpy().astype(np.uint8)
        transmute_times+=time()-t0

        for i in range(bunching):
            t0 = time()

            frame=cv2.cvtColor(grid_tens[i], cv2.COLOR_RGB2BGR)
            video.write(frame)
            record_times+=time()-t0

    
    # print('average step time : ',step_times)
    # print('average transmute time : ',transmute_times)
    # print('average record time : ',record_times)

    video.release()
    # saveTensVideo(tensor_out,folderpath='.',name=name,fps=fps,out_size=1500,columns=5)

if __name__=='__main__':
    import os

    columns = 5

    # batch_files = os.listdir('data/paper_search/batch')
    # batch_params = [os.path.join('data/paper_search/batch',f) for f in batch_files]
    location = 'data/random/batch'
    batch_files = os.listdir(location)
    batch_params = [os.path.join(location,f) for f in batch_files]
    save_name = 'rando_ln2'

    print('making a bunch of videos')
    for i,b_par in tqdm(enumerate(batch_params)):
        save_batch_video(b_par, batch_files[i].split('.')[0], save_name=save_name)