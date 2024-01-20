from batch_finder import param_generator
from modules.utils import b_finder_utils as f_utils



if __name__=='__main__':
    device = 'cuda:0'
    
    dir = 'data/random'
    dir_ind = 'data/random/individual'

    batch_size=20
    num= 5

    f_utils.save_rand(dir,batch_size,num,param_generator=param_generator,device=device)