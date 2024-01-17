import torch
import pygame
from Camera import Camera
from Automaton import *
from Write_video import *
import cv2
import pickle as pk
import os

import numpy as np


"""
    Plots and save video of the evolution of the automaton interpolating between two parameters sets.
    These are loaded from files.
    
    Like single_param_scan.py but with more parameters.

    <MAY NEED BUG FIXING>
"""

device = 'cuda'
W,H = 400,400
dt = 0.1

k = 1 # number of rings per kernel

steps = 10 # numbers of parameter sets to test
N_steps = 300 # number of steps the automaton goe through for each set


f_d = open("interesting/parameters/seed=7302773741657753678", 'rb')
params_d = { k:(v.to(device)) for k,v in (pk.load(f_d)).items() }
params_d['k_size'] = 25
f_d.close()


f_e = open("interesting/parameters/seed=14708293706067972719", 'rb')
params_e = { k:(v.to(device)) for k,v in (pk.load(f_e)).items() }
params_e['k_size'] = 25
f_e.close()

T = torch.linspace(0, 1, steps, device = device)

params = {}
params['k_size'] = 25

# for params of shape (3,3)
params['mu'] = (torch.ones_like(T[...,None,None], device = device)-T[...,None,None])*params_d['mu'][None] + T[...,None,None]*params_e['mu'][None]
params['sigma'] = (torch.ones_like(T[...,None,None], device = device)-T[...,None,None])*params_d['sigma'][None] + T[...,None,None]*params_e['sigma'][None]
params['weights'] = (torch.ones_like(T[...,None,None], device = device)-T[...,None,None])*params_d['weights'][None] + T[...,None,None]*params_e['weights'][None]

# for params of shape (3,3,k)
beta_d = params_d['beta'][None]
beta_e = params_e['beta'][None]
sigma_k_d = params_d['sigma_k'][None]
sigma_k_e = params_e['sigma_k'][None]
mu_k_d = params_d['mu_k'][None]
mu_k_e = params_e['mu_k'][None]

T = T[..., None, None, None]


params['beta'] = (torch.ones_like(T, device = device)-T)*beta_d + T*beta_e
params['sigma_k'] = (torch.ones_like(T, device = device)-T)*sigma_k_d + T*sigma_k_e
params['mu_k'] = (torch.ones_like(T, device = device)-T)*mu_k_d + T*mu_k_e  


# params_refined = {
#         'k_size' :params['k_size'],
#         'mu': params['mu'][2],
#         'sigma' : params['sigma'][2],
#         'beta' : params['beta'][2],
#         'mu_k' : params['mu_k'][2],
#         'sigma_k' : params['sigma_k'][2],
#         'weights' : params['weights'][2] # element i, j represents contribution from channel i to channel j
#     }

# f = open("/home/Zilan/Desktop/leniasearch/Archive/multi_scan/Clumps_2/parameters_refined", "wb") 
# pk.dump(params_refined,f)
# f.close() 


masses = []
videos = []



for i in range(steps):
    param = {
            'k_size' :params['k_size'],
            'mu': params['mu'][i],
            'sigma' : params['sigma'][i],
            'beta' : params['beta'][i],
            'mu_k' : params['mu_k'][i],
            'sigma_k' : params['sigma_k'][i],
            'weights' : params['weights'][i] # element i, j represents contribution from channel i to channel j
        }
        # Initialize the automaton
    auto = LeniaMC((W,H), dt, param,device=device)
    auto.to(device)
    videos.append(compute_video(auto, N_steps))





    # for j in range(N_steps):
    #     # Step the automaton if we are updating
    #     # THIS MIGHT BE CHANGED
    #     with torch.no_grad():
    #         auto.step()

    masses.append(auto.mass().sum()/480000.0)


plt.plot(range(steps), masses)
plt.suptitle('Mass evolution')
plt.xlabel('t')
plt.ylabel('Normalized mass')
plt.savefig('interesting\\refined_mass.jpg')
plt.show()

write_video(videos, 'interesting\\refined.mp4', 60.0)
