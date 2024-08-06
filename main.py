"""
    Scipt to run the automaton in real time. See README for a list of hotkeys.
"""
import torch
import pygame
from modules.Camera import Camera
from modules.Automaton import BatchLeniaMC, DiscreteLenia
from modules.utils.main_utils import compute_ker, load_params, around_params
import cv2
import pickle as pk
from batch_finder import param_generator
from modules.utils.b_finder_utils import param_batch_to_list
import numpy as np, os, random
from torchenhanced.util import showTens
#============================== PARAMETERS ==========================================================
device = 'cuda' # Device on which to run the automaton
W,H = 300,300 # Size of the automaton
dt = 0.1 # Time step size
num_channels= 1

interesting_dir = os.path.join('data','latest') # Directory containing the parameters to load when pressing 'm'
remarkable_dir = os.path.join('data','remarkable') # Directory containing the parameters to save when pressing 's'
#===========================DO NOT MODIFY BELOW THIS LINE===========================================

param_gen = lambda dev: param_generator(1,num_channels=num_channels,device=dev)


videos_dir = os.path.join('data','videos')

interesting_dir = os.path.join(interesting_dir,'individual')
remarkable_dir = os.path.join(remarkable_dir,'individual')
os.makedirs(interesting_dir, exist_ok=True)
os.makedirs(remarkable_dir, exist_ok=True)
os.makedirs(videos_dir, exist_ok=True)

interest_files = os.listdir(interesting_dir)

if len(interest_files) > 0:
    file = random.choice(interest_files)
    params = load_params(os.path.join(interesting_dir,file), make_batch=True, device=device)
else :
    params = param_gen(device)
    print('FUGG')

print(params['mu'].shape)   
# Initialize the automaton
# auto = BatchLeniaMC((1,H,W), dt, params=params, num_channels=num_channels, device=device)
auto = DiscreteLenia((1,H,W), discretization=13, params=None ,device=device)
auto.to(device)
# auto.update_params(params)
# kern = compute_ker(auto, device)

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
launch_video=True

n_steps = 0

frame= 0

chosen_interesting = 0

display_kernel = False

recording=False
launch_video = True

counter = 0 # counter to get only the frames we want

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
                # params = param_gen(device)
                params = auto.gen_batch_params(auto.device)
                auto.update_params(params)
                # kern = compute_ker(auto, device) 
            if(event.key == pygame.K_u):
                """ Variate around parameters"""
                params = around_params(params, device)
                auto.update_params(params)
                # kern = compute_ker(auto, device) 
            if(event.key == pygame.K_i):
                auto.set_init_fractal()
                n_steps=0
            if(event.key == pygame.K_j):
                auto.set_init_perlin(wavelength=40)
                n_steps=0
            if(event.key == pygame.K_k):
                sq_size = random.randint(5,min(W,H))
                auto.set_init_perlin(square_size=sq_size)
            if(event.key == pygame.K_m):
                n_steps=0
                # Load random interesting param
                file = interest_files[chosen_interesting]
                chosen_interesting = (chosen_interesting+1)%len(interest_files)

                print('loaded ', os.path.join(interesting_dir,file))
                params = load_params(os.path.join(interesting_dir,file),make_batch=True,device=device)

                auto.update_params(params)
                # kern = compute_ker(auto, device) 
            if(event.key == pygame.K_s):
                # Save the current parameters :
                para = auto.get_params()
                name = f'mu{para["mu"][0][0][0].item():.2f}_sigma{para["sigma"][0][0][0].item():.2f}_{para["beta"][0,0,0,0].item():.2f}'
                f = open(os.path.join(remarkable_dir,'int'+name+'.pk'), "wb") 
                pk.dump(para,f)
                f.close() 
            if(event.key == pygame.K_p):
                updating=not updating
            if(event.key == pygame.K_k):
                display_kernel = not display_kernel
            if(event.key == pygame.K_r):
                recording = not recording
                if(not launch_video):
                    video_out.release()
                    launch_video = True 
            if(event.key == pygame.K_DELETE):
                auto.state = torch.zeros_like(auto.state)

        # Handle the event loop for the camera
        camera.handle_event(event)
    
    if(updating):
        # Step the automaton if we are updating
        with torch.no_grad():
            auto.step()
            n_steps += 1

    auto.draw() # Draw the worldmap before retrieving it
    
    #Retrieve the world_state from automaton
    world_state = auto.worldmap
    # if display_kernel == True:
    #     world_state[:auto.k_size, auto.h-auto.k_size:auto.h,:] =  255*kern[0].cpu()
    #     world_state[auto.k_size:2*auto.k_size, auto.h-auto.k_size:auto.h,:] =  255*kern[1].cpu()  
    #     world_state[2*auto.k_size:3*auto.k_size, auto.h-auto.k_size:auto.h,:] =  255*kern[2].cpu()  

    #Make the viewable surface.
    surface = pygame.surfarray.make_surface(world_state)

    if(recording):
        if(launch_video):
            launch_video = False
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            para = auto.get_params()
            name = f'mu{para["mu"][0,0,0].item():.2f}_sigma{para["sigma"][0,0,0].item():.2f}'
            vid_loc = os.path.join(videos_dir,name+'.avi')
            # vid_loc = 'Videos/McLenia_orbiums_int.avi'
            video_out = cv2.VideoWriter(vid_loc, fourcc, 120.0, (W, H))
        if (counter%2 == 0):
            frame_bgr = cv2.cvtColor(auto.worldmap.transpose(1,0,2), cv2.COLOR_RGB2BGR)
            video_out.write(frame_bgr)
        pygame.draw.circle(surface, (255,0,0), (W-10,H-10),2)
        counter += 1
    
    
    # Clear the screen 
    screen.fill((0, 0, 0))

    m = [f"{x.item(): .2f}" for x in auto.mass().squeeze(0)]

    s = f'frames : {n_steps}, mass : {auto.mass().mean():.2f}'
    upMacro = font.render(s, False, (255,255,255), (0,0,0))
    

    # Draw the scaled surface on the window
    zoomed_surface = camera.apply(surface)

    screen.blit(zoomed_surface, (0,0))
    screen.blit(upMacro, (0,0))

    # Update the screen
    pygame.display.flip()

    clock.tick(120)  # limits FPS to 120


if(not launch_video):
    video_out.release()
pygame.quit()
