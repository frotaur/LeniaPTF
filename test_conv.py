import torch
from torchenhanced.util import showTens
from torch.nn.functional import conv2d

batch = 2
test_tensor = torch.zeros((batch,3,30,30))

test_tensor[0,0]=1
test_tensor[1,2]=1

test_tensor=test_tensor.reshape((1,6,30,30))
kernels = torch.zeros((batch,3,3,5,5))

kernels[0,0,2]=1

kernels[1,2,1]=1

kernels = kernels.reshape((9*batch,1,5,5))
out = conv2d(test_tensor, kernels,groups=3*batch) # (9*2,1,5,5)
out = out.reshape(batch,3,3,26,26) # (3,3,26,26)

showTens(out)