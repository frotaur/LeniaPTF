import torch
import pygame
from Camera import Camera
from Automaton import *
import cv2
import pickle as pk
import scipy as sp

import numpy as np

device = 'cuda'
W,H = 400,400
dt = 0.1

N_videos = 100 # numbers of parameter sets to test
N_steps = 300 # number of steps the automaton goe through for each set

def gen_params():
    mu = torch.rand((3,3), device=device)
    sigma = mu/(3*np.sqrt(2*np.log(2)))*(1+ (torch.ones_like(mu)-2*torch.rand_like(mu)))
        

    params = {
        'k_size' : 25, 
        'mu':  mu ,
        'sigma' : sigma,
        'beta' : torch.rand((3,3,1), device=device),
        'mu_k' : torch.rand((3,3,1), device=device),
        'sigma_k' : torch.rand((3,3,1), device=device),
        'weights' : torch.rand((3,3), device = device) # element i, j represents contribution from channel i to channel j
    }
    return params


for i in range(N_videos):
    params = gen_params()

    # Initialize the automaton
    auto = LeniaMC((W,H), dt, params,device=device)
    auto.to(device)

    for j in range(N_steps):
        # Step the automaton if we are updating
        # THIS MIGHT BE CHANGED
        with torch.no_grad():
            auto.step()

    if (max(auto.mass()) < 1e-6):
        # f = open("multi_sort/dies/seed="+str(torch.seed()), "wb") 
        # pk.dump(params,f)
        # f.close()
        params_d = params
        stop_d=False
        print("Found dead")  
    elif (max(auto.mass()) > 1e-6):
        # f = open("multi_sort/lives/seed="+str(torch.seed()), "wb") 
        # pk.dump(params,f)
        # f.close()
        params_e = params
        stop_a=False
        print("Found alive")