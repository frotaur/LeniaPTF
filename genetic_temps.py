from modules import GeneticEvolver
from modules import Ranker
from modules.reward_training import VideoRewardTrainer, CLIPVIPReward
import torch

model = VideoRewardTrainer.get_model_from_state(CLIPVIPReward, 'checkpoints/400_clipvip.state')
rewarder = VideoRewardTrainer(model=model, device='cuda:0')
ranker = Ranker(reward_trainer=rewarder, evolve_steps=800, device='cuda:0')

a = GeneticEvolver(ranker=ranker, rank_config='script_config/rank/classic.json', search_config='script_config/search/classic.json')

with torch.no_grad():
    params = a.evolve(pop_size=10, generations=10, elite_size=2, device='cuda:0')

params.save('evo.pt')