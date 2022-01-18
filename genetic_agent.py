import numpy as np
import ga
import csv
import matplotlib.pyplot as plt


def train_agent(num_generations, pop_size, new_population, num_parents_mating, num_mutations, max_lines_cleared=False):
    print("Beginning training")
    best_outputs = []
    data = []
    for generation in range(num_generations):
        print("Generation : ", generation+1)
        # Measuring the fitness of each chromosome in the population. TODO hier moet vgm elke ding in de population dus die game spelen en de fitness returnen
        fitness = ga.cal_pop_fitness(new_population,generation, num_generations, max_lines_cleared)
        print("Fitness")
        print(fitness)
        #
        best_outputs.append(np.max(fitness))
        # The best result in the current iteration.
        print("Best result : {}, Average : {}".format( np.max(fitness),np.average(fitness)))

        header = ['generation', 'score']
        data.append([generation+1, np.max(fitness)])
        with open('scores.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            writer.writerow(header)

            for x in data:
                # write the data
                writer.writerow(x)

        best_match_idx = np.where(fitness == np.max(fitness))
    # print(new_population)
        print("Best solution : ", new_population[best_match_idx, :])
        
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
        offspring_mutation = ga.mutation(offspring_crossover, num_mutations=num_mutations)
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
    # print(new_population)
    print("Best solution : ", new_population[best_match_idx, :])
    # print("Best solution fitness : ", fitness[best_match_idx])

    plt.plot(best_outputs)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.show()
    plt.savefig('plot.png')

#Creating the initial population.
num_weights = 99
max_lines_cleared = 20
sol_per_pop = 100
num_generations = 5
num_parents_mating = int(sol_per_pop*0.2) # Was 0.2% of best performing get to next generation
num_mutations = int(sol_per_pop*0.2)

# Defining the population size.
pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
new_population = np.random.uniform(low=-4.0, high=4.0, size=pop_size)

train_agent(num_generations, pop_size, new_population, num_parents_mating, num_mutations, max_lines_cleared)