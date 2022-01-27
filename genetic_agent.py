import numpy as np
import ga
import csv
import matplotlib.pyplot as plt
import pandas as pd
from numpy import average, savetxt
from numpy import loadtxt
import pickle
def train_agent(current_generation,num_generations,sol_per_pop ,  pop_size, new_population, num_parents_mating, num_mutations,mutate_percentage, isMultilayer, max_lines_cleared=False):
    print("Beginning training")
    best_outputs = []
    data = []
    for generation in range(current_generation,num_generations+num_generations):
        print("Generation : ", generation+1)
        # Measuring the fitness of each chromosome in the population. TODO hier moet vgm elke ding in de population dus die game spelen en de fitness returnen
        fitness = ga.cal_pop_fitness(new_population,generation, num_generations, max_lines_cleared)
        print("Fitness")
        print(fitness)
        #
        best_outputs.append(np.max(fitness))
        # The best result in the current iteration.
        print("Best result : {}, Average : {}".format( np.max(fitness),np.average(fitness)))


        best_match_idx = np.where(fitness == np.max(fitness))
    # print(new_population)
        print("Best solution : ", new_population[best_match_idx, :])
        #data.append([generation + 1, np.max(fitness), new_population[best_match_idx, :]])
        data.append([generation,np.max(fitness),np.average(fitness),new_population])
        if isMultilayer:
            output = open('data-multi-{}.pkl'.format(sol_per_pop), 'wb')
        else:
            output = open('data-single-{}.pkl'.format(sol_per_pop), 'wb')
        pickle.dump(data, output)
        output.close()

       
        # Selecting the best parents in the population for mating.
        parents = ga.select_mating_pool(new_population, fitness, 
                                        num_parents_mating)
        print((pop_size[0]-parents.shape[0], num_weights))
        # Generating next generation using crossover.
        offspring_crossover = ga.crossover(parents,
                                        offspring_size=(pop_size[0]-parents.shape[0], num_weights))

        # Adding some variations to the offspring using mutation.
        offspring_mutation = ga.mutation(offspring_crossover, mutate_percentage , num_mutations=num_mutations)
        # Creating the new population based on the parents and offspring.
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation
        
    # Getting the best solution after iterating finishing all generations.
    #At first, the fitness is calculated for each solution in the final generation.
    fitness = ga.cal_pop_fitness(new_population,-1,0,max_lines_cleared)
    # Then return the index of that solution corresponding to the best fitness.
    best_match_idx = np.where(fitness == np.max(fitness))
    print("Best solution : ", new_population[best_match_idx, :])

    plt.plot(best_outputs)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.savefig('plot.png')
    plt.show()


# User Options
isMultilayer = False
useSave = True

#Creating the initial population.
if isMultilayer:
    states = 9
    hidden_layers = 1
    num_weights = ((states+1)*9)*hidden_layers+ 9
else :
    num_weights = 9
# Game and Agent Options    
max_lines_cleared = 1
sol_per_pop = 100
num_generations = 5
num_parents_mating = int(sol_per_pop*0.2)
num_mutations = int(sol_per_pop*0.2)
mutate_percentage = 0.1
current_generation = 0
# Defining the population size.
pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.

def GET_best_weights(isMultilayer):
    if isMultilayer:
        pkl_file = open('data-multi-{}.pkl'.format(sol_per_pop), 'rb')
    else:
        pkl_file = open('data-single-{}.pkl'.format(sol_per_pop), 'rb')
    data = pickle.load(pkl_file)[-1]
    pkl_file.close()
    current_generation = data[0]
    max_fit = data[1]
    avg_fit = data[2]
    data = data[3]
    print('Generation: {}, Max fitness:{}, Average fitness: {}, number of mutations: {}'.format(current_generation+1, max_fit,avg_fit,len(data)))
    return data

if useSave:
    new_population = GET_best_weights(isMultilayer)
else:
    new_population = np.random.uniform(low=-4.0, high=4.0, size=pop_size)
train_agent(current_generation,num_generations, sol_per_pop, pop_size, new_population, num_parents_mating, num_mutations, mutate_percentage, isMultilayer, max_lines_cleared)

