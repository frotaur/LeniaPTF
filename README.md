# LENIA FINDER AND REAL-TIME SIMULATIONS

## How to use
First of all, install dependencies with `pip install -r requirements.txt`.

### Real-time simulation
To launch the simulation, run `main.py` with `python main.py`. Before doing that, you might want to change options. At the top of `main.py`, you can modify the following code :

```
device = 'cuda:0' # Device on which to run the automaton
W,H = 600,600 # Size of the automaton
dt = 0.1 # Time step size


interesting_dir = os.path.join('remark_record') # Directory containing the parameters to load when pressing 'm'
remarkable_dir = os.path.join('data','remarkable') # Directory containing the parameters to save when pressing 's'
```
Leave `interesting_dir` and `remarkable_dir` as default for now.

Run the main.py : `python main.py`. A window should open, that's the Lenia world. To make actions, use the following hotkeys :

- `n` : Changes to random new parameters. Sampled from `param_generator` in `batch_finder.py`
- `m` : Load random phase transition parameters (or any saved parameters in `interesting_dir`)
- `u` : Slightly vary around the existing parameters. Can be pressed repeatedly.
- `j` : Re-initialize state with Perlin noise
- `s` : Save the current parameters inside `remarkable_dir`
- `p` : Pause/unpause simulation
- `k` : Display current 
- `r` : start/stop recording video. Video will be saved in `./videos` when closing main window.
- `delete` : set the state to the dead state.

To explore remarkable parameters we found, launch the main.py, press `m` and subsequently `j` to seed the initial condition. Behold the creatures living in your computer ! Each time you change parameters with `m`, do not forget to reseed the initial condition with `j`. Use `n` to explore parameters sampled randomly from the prior region defined in `param_generator`.

### Generate phase transition parameters
The generation of phase transition parameters using PTF is done with `batch_finder.py`. Note that for the generation to not be too slow, it is recommended to have a GPU for this.

The parameters than can be altered are present at the beginning of the file :

```
#============================== PARAMETERS ==========================================================

# Where to save the found parameters
folder_save = './data/paper_search'

device = 'cuda:0'
H,W = 200,200 # Size of the automaton
dt = 0.1 # Time step size
N_steps = 800 # Number of steps to run the automaton for

num_points = 40 # Number of points to find
refinement = 8 # Number of steps to run the dichotomy search for
cross=False # If True, will compute the transition point between all pairs of parameters. Useful for huge generations, but lessens variations
use_mean = True # If True, uses the mean of the activations to determine death. If False, uses the max.

# threshold below which we say we have found a dead config in the initial search
threshold_e = 0.05
# threshold below which we say we have found a dead config in the dichotomy search (generally matches threshold_e)
threshold_i = 0.05

batch_size = 20 # Number of worlds to simulate in parallel. Reduce if you run out of memory

# Uncomment to use the equivalent of a 'TEMP' directory. IS EMPTIED EACH TIME THE SCRIPT IS RUN
folder_save= 'data/latest'
```
Important parameters to note are :
- `folder_save` : folder location in which to save the found parameters. Note that parameters will be saved as duplicate, once in `folder_save/batch` as 'batched' parameters, and once in `folder_save/individual` as 'individual' (unbatched) parameters.
- `device`: set this to 'cpu' if you don't have a GPU, otherwise to the ordinal of your GPU ('cuda:0' if only one GPU)
- `N_steps` : number of steps to run the automaton before determining the phase.
- `num_points` : total number of phase transition point to generate
- `batch_size` : Number of world to generate in parallel. If the program crashes with a CUDA error, lower this.

Not the last line; if this is left uncommented, it will save the parameters in `data/latest` instead of the specified folder. Each time `batch_finder.py` is run, this folder is emptied. Useful to try a lot of different generations !

Finally, the function `param_generator` is defined below. This function defines the prior that generates the parameters. It can be adjusted if you want to change a prior. Be sure to return a dictionary containing tensor in the same sizes as presented.

All ready ! All you need to do is to run `python batch_finder.py`, which will run the PTF algorithm with the above specifications. To see real-time simulation of the discovered parameters, head into `main.py`, and change `interesting_dir` to the value of `folder_save` in `batch_finder`. Now, when pressing `m`, you will see the dynamics for the parameters you just generated !

Note : the random prior parameters are also save in `'data/latest_rand'`.
### Generate videos
To generate videos, you can either use the `r` key in main.py and record live simulation, or you can use the file `indiv_video.py`. The parameters are as follows :

```
param_dir = 'data/latest_rand/individual' # Directory containing the individual (unbatched) parameters
out_dir = 'test_videos' # Directory to save the videos


simulation_time = 1800 # Number of frames to simulate
size = 500,500 # Size of the simulation
fps=120 # Framerate of the video
device='cuda:0' # Device on which to simulate
```

Most are self-explanatory. Set `param_dir` to the folder containing a set of **individual** parameters, choose the output directory, and run the script. It should generate videos of the dynamics. It takes a while to run !