import torch
import pygame
from Camera import Camera
from Automaton import *
import cv2
import pickle as pk
import os

import numpy as np


def extremum_finder(W, H, dt, N_steps, params_range, threshold,  device):
    """
        Finds a set of parameter of dead automaton, and a set of parameters of an alive automaton.
    """

    ## Parameters to scan over

    stop_d = True
    stop_a = True

    while (stop_a or stop_d):
        params = params_range()
        # Initialize the automaton
        auto = LeniaMC((W,H), dt, params, device=device)
        auto.to(device)

        for j in range(N_steps):
            # Step the automaton if we are updating
            # THIS MIGHT BE CHANGED
            with torch.no_grad():
                auto.step()


        if ((max(auto.mean_mass()) < threshold) and stop_d): 
            # f = open("multi_sort/dies/seed="+str(torch.seed()), "wb") 
            # pk.dump(params,f)
            # f.close()
            params_d = params
            stop_d=False
            print("Found dead")  
        elif ((max(auto.mean_mass()) > threshold) and stop_a):
            # f = open("multi_sort/lives/seed="+str(torch.seed()), "wb") 
            # pk.dump(params,f)
            # f.close()
            params_e = params
            stop_a=False
            print("Found alive")
    
    return params_d, params_e
       

@torch.no_grad()
def interest_finder(W,H, dt, N_steps, params_d, params_a, refinement, threshold, device):
    """
        By dichotomy, finds the parameters of an interesting automaton. By interesting, here
        we mean a set of parameters which lies at the transition between an asymptotically dead
        and an asymptotically alive automaton.

        Args :
            W, H : width and height of the automaton 
            dt : time step
            N_steps : number of steps the automaton goes through for each set of parameters
            device : device on which to run the automaton
            params_d : parameters of a dead automaton   (dict)
            params_a : parameters of an alive automaton  (dict)
            refinement : number of iterations of dichotomy
    """
    p1 = params_d
    p2 = params_a
    t_crit = 0.5

    for i in range(refinement):

        params = {
            'k_size' : 25, 
            'mu':  (p1['mu']+p2['mu'])/2 ,
            'sigma' : (p1['sigma']+p2['sigma'])/2 ,
            'beta' : (p1['beta']+p2['beta'])/2,
            'mu_k' : (p1['mu_k']+p2['mu_k'])/2,
            'sigma_k' : (p1['sigma_k']+p2['sigma_k'])/2,
            'weights' : (p1['weights']+p2['weights'])/2 # element i, j represents contribution from channel i to channel j
        }

        auto = LeniaMC((W,H), dt, params ,device=device)
        auto.to(device)

        for _ in range(N_steps):
            # Step to get to the 'final' state
            auto.step()
        
        if (max(auto.mean_mass()) < threshold): # Also put this value as parameter, and the same in both ifs
            p1 = params
            t_crit += 0.5**(i+2)
        elif(max(auto.mean_mass()) > threshold):
            p2 = params
            t_crit -= 0.5**(i+2)
    #print("Interest found !")

    return t_crit
              