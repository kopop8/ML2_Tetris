
import numpy as np
import ga
"""
The y=target is to maximize this equation ASAP:
    y = w1x1+w2x2+w3x3+w4x4+w5x5+6wx6
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7)
    What are the best values for the 6 weights w1 to w6?
    We are going to use the genetic algorithm for the best possible values after a number of generations.
"""



"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
# Number of the weights we are looking to optimize.


#Creating the initial population.
num_weights = 9
sol_per_pop = 50
num_parents_mating = 2 # Was 4

# Defining the population size.
pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
new_population = np.random.uniform(low=-4.0, high=4.0, size=pop_size)
print("Beginning training")
best_outputs = []
num_generations = 10
for generation in range(num_generations):
    print("Generation : ", generation+1)
    # Measuring the fitness of each chromosome in the population. TODO hier moet vgm elke ding in de population dus die game spelen en de fitness returnen
    fitness = ga.cal_pop_fitness(new_population,generation, num_generations)
    print("Fitness")
    print(fitness)
    #
    best_outputs.append(np.max(fitness))
    # The best result in the current iteration.
    print("Best result : {}, Average : {}".format( np.max(fitness),np.average(fitness)))
    
    # Selecting the best parents in the population for mating.
    parents = ga.select_mating_pool(new_population, fitness, 
                                      num_parents_mating)
    # print("Parents")
    # print(parents)

    # Generating next generation using crossover.
    offspring_crossover = ga.crossover(parents,
                                       offspring_size=(pop_size[0]-parents.shape[0], num_weights))
    # print("Crossover")
    # print(offspring_crossover)

    # Adding some variations to the offspring using mutation.
    offspring_mutation = ga.mutation(offspring_crossover, num_mutations=2)
    # print("Mutation")
    # print(offspring_mutation)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation
    
# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
fitness = ga.cal_pop_fitness(new_population,-1,0)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = np.where(fitness == np.max(fitness))
print(new_population)
print("Best solution : ", new_population[best_match_idx, :])
# print("Best solution fitness : ", fitness[best_match_idx])


import matplotlib.pyplot
matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()
