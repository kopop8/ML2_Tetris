import numpy as np
import ga
import csv
import matplotlib.pyplot as plt


def train_agent(num_generations, pop_size, new_population, num_parents_mating, num_mutations,mutate_percentage, max_lines_cleared=False):
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



        best_match_idx = np.where(fitness == np.max(fitness))
    # print(new_population)
        print("Best solution : ", new_population[best_match_idx, :])



        try:
            data = []
            with open('scores.csv', 'r') as f:
                csv_reader = csv.reader(f)
                for line in csv_reader:
                    if line != '':
                        data.append(line)
            #data.append([generation + 1, np.max(fitness), new_population[best_match_idx, :]])
            data.append([generation + 1, np.max(fitness), new_population])
            list2 = [x for x in data if x != []]
            with open('scores.csv', 'w', encoding='UTF8') as f:
                writer = csv.writer(f)

                for x in list2:
                    # write the data
                    writer.writerow(x)
        except:
            print("File does not exist, so create file with ")
            header = ['generation', 'score', 'weights']
            #data.append([generation + 1, np.max(fitness), new_population[best_match_idx, :]])
            data.append([generation + 1, np.max(fitness), new_population])
            list2 = [x for x in data if x != []]

            with open('scores.csv', 'w', encoding='UTF8') as f:
                writer = csv.writer(f)

                writer.writerow(header)

                for x in list2:
                    # write the data
                    writer.writerow(x)
        
        # Selecting the best parents in the population for mating.
        parents = ga.select_mating_pool(new_population, fitness, 
                                        num_parents_mating)
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


#Creating the initial population.
num_weights = 99
max_lines_cleared = 20
sol_per_pop = 5
num_generations = 5
num_parents_mating = int(sol_per_pop*0.2)
num_mutations = int(sol_per_pop*0.2)
mutate_percentage = 0.5

# Defining the population size.
pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.

def GET_best_weights():
    with open('scores.csv', "r") as f1:
        data = []
        csv_reader = csv.reader(f1)
        for line in csv_reader:
            data.append(line)
        array = data[-2][2]
        array = array.replace('  ', ',')
        array = array.replace(' -', ',-')
        return array

#print(np.random.uniform(low=-4.0, high=4.0, size=pop_size))

try:
    data = GET_best_weights()
    del data[0]
    del data[-1]
    new_population = [data]
except:
    new_population = np.random.uniform(low=-4.0, high=4.0, size=pop_size)

train_agent(num_generations, pop_size, new_population, num_parents_mating, num_mutations, mutate_percentage, max_lines_cleared)