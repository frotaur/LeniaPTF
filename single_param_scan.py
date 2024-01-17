<<<<<<< HEAD
import torch
import pygame
from Camera import Camera
from Automaton import *
from Write_video import *
import cv2
import pickle as pk
import os

import numpy as np




device = 'cuda'
W,H = 300,300
dt = 0.1

steps = 10 # numbers of parameter sets to test
N_steps = 50 # number of steps the automaton goe through for each set


f_d = open("Parameters\dies\seed=1135423127701300", 'rb')
params_d = pk.load(f_d)
f_d.close()


f_e = open("Parameters\seed=1135551540039400", 'rb')
params_e = pk.load(f_e)
f_e.close()


mu_d, sigma_d = params_d['mu'][0][0].item(), params_d['sigma'][0][0].item(),
mu_e, sigma_e = params_e['mu'][0][0].item(), params_e['sigma'][0][0].item(),

# mu_d, sigma_d = params_d['mu'][0][0].item(), params_d['sigma'][0][0].item(),
# mu_e, sigma_e = 0.5, 0.5

print('[' +  str(mu_d) + ',' + str(sigma_d) + ']->['+  str(mu_e) + ',' + str(sigma_e) + ']')

sigma = torch.linspace(sigma_d, sigma_e, steps, device=device)[...,None,None,None] # (steps, 1,1,1)
mu = torch.linspace(mu_d, mu_e, steps, device = device)[...,None,None,None] # (steps, 1,1,1)
print(mu.shape)


P = torch.cat((mu, sigma), dim=1) # (steps, 2,1,1)
P = P.expand(-1, -1, 3, 3) # (steps, 2,3,3)
print(P.shape)

masses = []
videos = []

for i in range(steps):
    params = {
            'k_size' : 25,
            'mu': P[i][0],
            'sigma' :P[i][1],
            'beta' : params_d['beta'],
            'mu_k' : params_d['mu_k'],
            'sigma_k' : params_d['sigma_k'],
            'weights' : params_d['weights'] # element i, j represents contribution from channel i to channel j
        }
        # Initialize the automaton
    auto = LeniaMC((W,H), dt, params,device=device)
    auto.to(device)
    videos.append(compute_video(auto, N_steps))





    # for j in range(N_steps):
    #     # Step the automaton if we are updating
    #     # THIS MIGHT BE CHANGED
    #     with torch.no_grad():
    #         auto.step()

    masses.append(auto.mass().sum()/270000.0)


plt.plot(range(steps), masses)
plt.suptitle('Mass evolution : (mu, sigma) = ' + '[' +  str(mu_d) + ',' + str(sigma_d) + ']->['+  str(mu_e) + ',' + str(sigma_e) + ']')
plt.xlabel('t')
plt.ylabel('Normalized mass')
plt.savefig('Fig_scan\\[' +  str(mu_d) + ',' + str(sigma_d) + ']-['+  str(mu_e) + ',' + str(sigma_e) + '].jpg')
plt.show()

=======
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
    Plots and save video of the evolution of the automaton interpolating between two parameters sets
"""


device = 'cuda'
W,H = 300,300
dt = 0.1

steps = 10 # numbers of parameter sets to test
N_steps = 50 # number of steps the automaton goe through for each set


f_d = open("Parameters\dies\seed=1135423127701300", 'rb')
params_d = pk.load(f_d)
f_d.close()


f_e = open("Parameters\seed=1135551540039400", 'rb')
params_e = pk.load(f_e)
f_e.close()


mu_d, sigma_d = params_d['mu'][0][0].item(), params_d['sigma'][0][0].item(),
mu_e, sigma_e = params_e['mu'][0][0].item(), params_e['sigma'][0][0].item(),

# mu_d, sigma_d = params_d['mu'][0][0].item(), params_d['sigma'][0][0].item(),
# mu_e, sigma_e = 0.5, 0.5

print('[' +  str(mu_d) + ',' + str(sigma_d) + ']->['+  str(mu_e) + ',' + str(sigma_e) + ']')

sigma = torch.linspace(sigma_d, sigma_e, steps, device=device)[...,None,None,None] # (steps, 1,1,1)
mu = torch.linspace(mu_d, mu_e, steps, device = device)[...,None,None,None] # (steps, 1,1,1)
print(mu.shape)


P = torch.cat((mu, sigma), dim=1) # (steps, 2,1,1)
P = P.expand(-1, -1, 3, 3) # (steps, 2,3,3)
print(P.shape)

masses = []
videos = []

for i in range(steps):
    params = {
            'k_size' : 25,
            'mu': P[i][0],
            'sigma' :P[i][1],
            'beta' : params_d['beta'],
            'mu_k' : params_d['mu_k'],
            'sigma_k' : params_d['sigma_k'],
            'weights' : params_d['weights'] # element i, j represents contribution from channel i to channel j
        }
        # Initialize the automaton
    auto = LeniaMC((W,H), dt, params,device=device)
    auto.to(device)
    videos.append(compute_video(auto, N_steps))





    # for j in range(N_steps):
    #     # Step the automaton if we are updating
    #     # THIS MIGHT BE CHANGED
    #     with torch.no_grad():
    #         auto.step()

    masses.append(auto.mass().sum()/270000.0)


plt.plot(range(steps), masses)
plt.suptitle('Mass evolution : (mu, sigma) = ' + '[' +  str(mu_d) + ',' + str(sigma_d) + ']->['+  str(mu_e) + ',' + str(sigma_e) + ']')
plt.xlabel('t')
plt.ylabel('Normalized mass')
plt.savefig('Fig_scan\\[' +  str(mu_d) + ',' + str(sigma_d) + ']-['+  str(mu_e) + ',' + str(sigma_e) + '].jpg')
plt.show()

>>>>>>> VassRefactor
write_video(videos, 'Videos\\[' +  str(mu_d) + ',' + str(sigma_d) + ']-['+  str(mu_e) + ',' + str(sigma_e) + '].mp4', 30.0)