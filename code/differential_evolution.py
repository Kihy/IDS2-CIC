from topology.traffic import generate_position
import numpy as np
from pyswarms.backend.handlers import BoundaryHandler
from tqdm import tqdm

def differential_evolution(n_dims, evolutions, f, args, bounds, original_time):

    population=20
    pbar=tqdm(range(evolutions),position=1)
    discrete_index=1
    #initialization
    positions=generate_position(population, n_dims, bounds, discrete_index)

    # original point
    positions[0][1]=0
    positions[0][0]=original_time
    
    position_history=[]
    position_history.append(positions[:,1:])
    bh=BoundaryHandler(strategy="nearest")
    costs, aux = f(positions, **args) # Compute current cost
    for e in range(evolutions):

        # evaluation
        trial_pop=[]
        #mutation
        for i in range(population):
            mutation_candidate = [idx for idx in range(population) if idx != i]
            mutation_candidate=np.random.choice(mutation_candidate, 3, replace=False)
            a,b,c=positions[mutation_candidate]
            mutation_factor=0.8
            mutant=a+mutation_factor*(b-c)
            #recombination
            crossp=0.7
            cross_points = np.random.rand(n_dims) < crossp
            trial = np.where(cross_points, mutant, positions[i])

            trial_pop.append(trial)

        trial_pop=np.array(trial_pop)
        # round and bound population
        trial_pop[:,discrete_index:]=np.rint(trial_pop[:,discrete_index:])
        trial_pop = bh(trial_pop, bounds)
        #evaluation

        trial_costs, trial_aux = f(trial_pop, **args)

        mask_cost = trial_costs < costs
        mask_pos = np.expand_dims(mask_cost ,axis=1)
        # update new positons Apply masks
        positions = np.where(~mask_pos, positions, trial_pop)
        costs = np.where(~mask_cost, costs, trial_costs)

        aux = np.where(~mask_pos, aux, trial_aux)
        position_history.append(positions[:,1:])
        post_fix="c: {:.4f}".format(np.min(costs))
        pbar.set_postfix_str(post_fix)
        pbar.update(1)

    best_index=np.argmin(costs)
    best_cost=costs[best_index]
    best_position=positions[best_index]
    best_aux=aux[best_index]
    std=np.std(positions, axis=0)
    return best_cost, best_position, best_aux, std,position_history
    #
    # # record the iteration with best cost
    # new_pbest_iter=np.where(
    #     ~mask_cost, swarm.pbest_iter, iter
    # )
