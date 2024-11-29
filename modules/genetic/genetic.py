import torch
from ..Automaton import BatchLeniaMC
from ..reward_training import VideoRewardTrainer
from ..utils.finder_utils import search_transition
from ..utils import LeniaParams

class GeneticEvolver:
    """
        Class to run genetic algorithms with Lenia.
        For now, it uses the 'ranker' class, as well as the specific automaton.
        In the future, it might be better to create wrapper classes for the scorer, and
        the automaton, we will see if it's necessary.
    """

    def __init__(self, ranker, rank_config, search_config):
        """
            Args:
                ranker : Ranker, the ranker object to use
                rank_config : dict, configuration for the ranker
        """
        self.ranker = ranker
        self.rank_config = rank_config
        self.search_config = search_config

        self.simulator = self.ranker.get_simulator(self.rank_config)
    
    def _init_population(self, pop_size, device='cpu'):
        """
            Initializes the population.

            Args:
                pop_size : int, size of the population
                device : str, device to use
        """
        # Generate random parameters
        params = LeniaParams(k_size=self.search_config['k_size'], batch_size=pop_size, device=device)

        return params

    def evolve(self, pop_size, generations, elite_size, device = 'cpu'):
        """
            Runs the genetic algorithm.

            Args:
                pop_size : int, size of the population
                generations : int, number of generations to run
                elite_size : int, number of elite individuals to keep
                ### ADD PARAMS TO CLASS OR TO THIS FUNCTION TO CONTROL MUTATION AND CROSSOVER
                num_channels : int, number of channels in the automaton
                device : str, device to use
        """
        elite_fraction = elite_size / pop_size
        mutation_needed = (pop_size - elite_size)//elite_size+1
        # Initialize the population
        population: LeniaParams = self._init_population(pop_size, device) # Those are parameters

        for gen in range(generations):
            # Score the population
            scores = self.ranker.score_params(population, self.simulator, repetitions=1) # (pop_size,)

            best, best_idx = torch.topk(scores, k=elite_size, largest=True, sorted=True)

            # Keep the elite
            elite = population[best_idx]

            # Mutate
            offspring = elite.mutate(rate=0.8)
            for _ in range(mutation_needed-1):
                offspring = offspring.cat(elite.mutate(rate=0.8))
            offspring = offspring[:pop_size-elite_size]
            # Combine the elite
            population = elite.cat(offspring)