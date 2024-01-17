from main_utils import gen_batch_params,gen_params
from Automaton import BatchLeniaMC,LeniaMC
import cv2
import torch, numpy as np


batch_size = 10
time = 100
output_file = 'output.avi'
fps=60

paramu = gen_batch_params(batch_size=batch_size,device= 'cuda:0')

auto = BatchLeniaMC((batch_size,300,300),0.1,paramu,device='cuda:0')

auto.set_init_perlin()

T, C, H, W = time,3, 300, 300
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(output_file, fourcc, fps, (W, H))

tensor_out = torch.zeros((batch_size,T,C,H,W),dtype=torch.float32,device='cuda:0')
for i in range(time):
    auto.step()
    tensor_out[:,i,:,:,:] = auto.state

tensor_out = tensor_out.reshape(batch_size*T,C,H,W).cpu().numpy()

for t in range(T*batch_size):
    frame = (tensor_out[t] * 255).astype(np.uint8)
    frame = frame.transpose(1, 2, 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video.write(frame)

video.release()

# Example usage
# tensor = your_tensor_here
# make_video(tensor)