from modules import Ranker
from modules.reward_training import VideoRewardTrainer,CLIPVIPReward
from torchenhanced import Trainer
from finder_script import param_generator

def load_model_from_state(constructor, state_file):
    namu,config,state_dict = Trainer.model_config_from_state(state_file,device='cpu')
    model = constructor(**config)
    model.load_state_dict(state_dict, strict=True)

    return model

model = load_model_from_state(CLIPVIPReward,'reward_checkpoints/400_clipvip.state')
rewarder = VideoRewardTrainer(model=model,device='cuda:0')

r = Ranker(reward_trainer=rewarder, evolve_steps=800, device='cuda:0')


r.generate_interesting_params(output_folder='YAAY', score_cutoff=2.,param_generator=param_generator, search_params='script_config/rank/classic.json',
                              use_ptf=False)