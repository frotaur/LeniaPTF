import torch
import pygame
from Camera import Camera
from Automaton import *
from main_utils import *
import cv2
import pickle as pk
import scipy as sp

import numpy as np, os, random

device = 'cuda:0'
W,H = 300,300
dt = 0.1

interesting_dir = 'data\interesting'
remarkable_dir = 'data\remarkable'
videos_dir = 'data\videos'

os.makedirs(interesting_dir, exist_ok=True)
os.makedirs(remarkable_dir, exist_ok=True)
os.makedirs(videos_dir, exist_ok=True)

interest_files = os.listdir(interesting_dir)

file = random.choice(interest_files)
p = load_params(os.path.join(interesting_dir,file), device=device)
params = p[0]
if p[1]:
    params_d = p[2]
    params_a = p[3]
    t_crit = p[4]


# Initialize the automaton
auto = LeniaMC((W,H), dt, params, device=device)
auto.to(device)
kern = compute_ker(auto, device)

print(auto.device)

# Initialize the pygame screen 
pygame.init()
font = pygame.font.SysFont('consolas',10)

screen = pygame.display.set_mode((W,H),flags=pygame.SCALED | pygame.RESIZABLE)
clock = pygame.time.Clock()

running = True
camera = Camera(W,H)

#Initialize the world_state array, of size (W,H,3) of RGB values at each position.
world_state = np.random.randint(0,255,(W,H,3),dtype=np.uint8)

updating = True

Red = False
Green = False
Blue = False
i = torch.tensor([0,1,2])

n_steps = 0

launch_video=True
recording = False

x,y = torch.meshgrid(torch.arange(0,W,device=device),torch.arange(0,H,device=device))
left_dragging = False
right_dragging = False
paint_size = 2
frame= 0

chosen_interesting = 0

display_kernel = False


#auto.update_params(load_params("ParamLoop\seed=9010333850255899677"))
while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        # Event loop. Here we deal with all the interactivity
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if(event.key == pygame.K_n):
                """ New random parameters"""
                params = gen_params(device)
                auto.update_params(params)
            if(event.key == pygame.K_u):
                """ Variate around parameters"""
                params = around_params(params, device)
                auto.update_params(params)
                kern = compute_ker(auto, device) 
            if(event.key == pygame.K_i):
                auto.set_init_simplex()
                n_steps=0
            if(event.key == pygame.K_m):
                n_steps=0
                # Load random interesting param
                file = interest_files[chosen_interesting]
                chosen_interesting = (chosen_interesting+1)%len(interest_files)

                print('loaded ', os.path.join(interesting_dir,file))
                p = load_params(os.path.join(interesting_dir,file))
                params = p[0]
                if p[1]:
                    params_d = p[2]
                    params_a = p[3]
                    t_crit = p[4]
                auto.update_params(params)
                kern = compute_ker(auto, device) 
            if(event.key == pygame.K_s):
                # Save the current parameters :
                para = auto.get_params()
                name = f'mu{para["mu"][0][0].item():.2f}_sigma{para["sigma"][0][0].item():.2f}_{para["beta"][0,0,0].item():.2f}'
                f = open(os.path.join(remarkable_dir,'int'+name+'.pk'), "wb") 
                pk.dump(para,f)
                f.close() 
            if(event.key == pygame.K_p):
                updating=not updating
            if(event.key == pygame.K_RIGHT):
                if p[1]:
                    t_crit -= 0.01
                    params = sum_params(params_a, params_d, t_crit)
                    auto.update_params(params)
            if(event.key == pygame.K_LEFT):
                if p[1]:
                    t_crit += 0.01
                    params =  sum_params(params_a, params_d, t_crit)
                    auto.update_params(params)
            if(event.key == pygame.K_k):
                display_kernel = not display_kernel
            if(event.key == pygame.K_a):
                release_video = True
                recording = not recording
            if(event.key == pygame.K_DELETE):
                auto.state = torch.zeros_like(auto.state)
            # if(event.key == pygame.K_DOWN):
            #     paint_size-=0.1
            #     paint_size=max(0,paint_size)
            # if(event.key ==pygame.K_UP):
            #     paint_size+=0.1
            # if(event.key==pygame.K_r):
            #     Red = True
            #     Green = False
            #     Blue = False
            # if(event.key==pygame.K_g):
            #     Green = True
            #     Red = False
            #     Blue = False
            # if(event.key==pygame.K_b):
            #     Blue = True
            #     Red = False
            #     Green = False
        # if event.type == pygame.MOUSEBUTTONDOWN :
        #         if event.button == pygame.BUTTON_LEFT:  # If left mouse button pressed
        #             left_dragging = True
        #         if event.button == pygame.BUTTON_RIGHT :
        #             right_dragging = True
        # elif event.type == pygame.MOUSEBUTTONUP:
        #     if event.button == pygame.BUTTON_LEFT:  # If left mouse button released
        #         left_dragging = False
        #     if event.button == pygame.BUTTON_RIGHT :
        #         right_dragging = False
        # elif event.type == pygame.MOUSEMOTION :
        #     if(left_dragging or right_dragging):
        #         (w,h) = event.pos
        #         dist = ((x-h)**2+(y-w)**2).to(torch.float)
        #     if(left_dragging):
        #         t = torch.where(dist<paint_size*((auto.k_size-1)),0.5*torch.rand_like(dist,device=device),0)
        #         t=t[None].repeat(3,1,1)
        #         if Red:
        #             t[1] = torch.zeros_like(t[1])
        #             t[2] = torch.zeros_like(t[2])
        #             auto.state=torch.clamp(auto.state+t,0,1)
        #         if Green:
        #             t[2] = torch.zeros_like(t[2])
        #             t[0] = torch.zeros_like(t[0])
        #             auto.state=torch.clamp(auto.state+t,0,1)
        #         if Blue:
        #             t[1] = torch.zeros_like(t[1])
        #             t[0] = torch.zeros_like(t[0])
        #             auto.state=torch.clamp(auto.state+t,0,1)
        #     if(right_dragging):
        #         auto.state=torch.where(dist<paint_size*((auto.k_size-1)),0,auto.state)

        # Handle the event loop for the camera
        camera.handle_event(event)
    
    if(updating):
        # Step the automaton if we are updating
        with torch.no_grad():
            auto.step()
            n_steps += 1

    auto.draw() # Draw the worldmap before retrieving it
    #Retrieve the world_state from automaton
    
    #Retrieve the world_state from automaton
    world_state = auto.worldmap
    if display_kernel == True:
        world_state[:auto.k_size, auto.h-auto.k_size:auto.h,:] =  255*kern[0].cpu()
        world_state[auto.k_size:2*auto.k_size, auto.h-auto.k_size:auto.h,:] =  255*kern[1].cpu()  
        world_state[2*auto.k_size:3*auto.k_size, auto.h-auto.k_size:auto.h,:] =  255*kern  [2].cpu()  
        

    #Make the viewable surface.
    surface = pygame.surfarray.make_surface(world_state)
    
    if(recording):
        if(launch_video):
            launch_video = False
            fourcc = cv2.VideoWriter_fourcc(*'HFYU')
            para = auto.get_params()
            name = f'mu{para["mu"][0][0].item():.2f}_sigma{para["sigma"][0][0].item():.2f}'
            vid_loc = os.path.join(videos_dir,name+'.avi')
            # vid_loc = 'Videos/McLenia_orbiums_int.avi'
            video_out = cv2.VideoWriter(vid_loc, fourcc, 60.0, (W, H))

        frame_bgr = cv2.cvtColor(auto.worldmap, cv2.COLOR_RGB2BGR)
        video_out.write(frame_bgr)
        pygame.draw.circle(surface, (255,0,0), (W-10,H-10),2)
    # Clear the screen 
    screen.fill((0, 0, 0))

    m = [f"{x.item(): .2f}" for x in auto.mean_mass()]
    # cent = auto.centroid()
    # print(cent[0])
    # cent = [f"({y[0].item(): .2f}, {y[1].item(): .2f})" for y in auto.centroid().transpose(0,1)]  
    # cent = [f"sig : {float(auto.sigma_k[random.randint(0,2),random.randint(0,2),random.randint(0,2)]):.2f} "]
    # s = " ".join(m) + " " + " ".join(cent)
    # s = str(erf[0])+str(erf[1])+str(erf[2]) + '\n'+str(erf[3])+str(erf[4])+str(erf[5]) + '\n'+str(erf[6])+str(erf[7])+str(erf[8])
    # upMacro = font.render(s, False, (255,255,255), (0,0,0))
    

    # Draw the scaled surface on the window
    zoomed_surface = camera.apply(surface)

    screen.blit(zoomed_surface, (0,0))
    # screen.blit(upMacro, (0,0))

    # Update the screen
    pygame.display.flip()
    # flip() the display to put your work on screen

    clock.tick(120)  # limits FPS to 60


if(not launch_video):
    video_out.release()
pygame.quit()
