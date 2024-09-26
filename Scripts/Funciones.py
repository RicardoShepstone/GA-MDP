
"""
@author: Ricardo Ignacio Shepstone Aramburu

"""
import math
import numpy as np
import random

def read_distances(file_name: str):
    distances = {}
    with open(file_name,'r') as file:
        n,m = map(int, file.readline().split())
        for line in file:
            element1, element2, distance = map(float, line.split())
            element1, element2 = int(element1), int(element2)
            distances[(element1,element2)] = distance
    return distances


def calculate_fitness(chromosome, dist):
    #dist_list = [dist[(i,j)] for i in genome for j in genome[genome.index(i)+1:]]
    
    dist_list = []
    for i in chromosome:
        for j in chromosome[chromosome.index(i)+1:]:
            if j>i:
                dist_list.append(dist[(i,j)])
            else:
                dist_list.append(dist[(j,i)])
                
    return math.fsum(dist_list)

def evaluate_population(population, dist):
    fitness_list = []
    for chromosome in population:
        fitness_list.append(calculate_fitness(chromosome, dist))
    return fitness_list

def generate_chromosome(m,n):
    return list(np.random.choice(n, size = m, replace = False))

def generate_population(population_size: int):
    return [generate_chromosome(50,500) for p in range(population_size)]

def crossover_operation(parent1, parent2):
    common_elements = list(set(parent1).intersection(set(parent2)))
    for element in common_elements:
        parent1.remove(element)
        parent2.remove(element)
    
    crossover_point = random.randint(1,len(parent1)-1)
    offspring1 = parent1[0:crossover_point]+common_elements+parent2[crossover_point:]
    offspring2 = parent2[0:crossover_point]+common_elements+parent1[crossover_point:]
    return offspring1, offspring2
    
def mutation_operation(chromosome, pm):
    mutated_chromosome = chromosome.copy()
    for i in range(len(chromosome)):
        if random.random()<pm:
            while True:
                mutation = random.randint(0,499)
                print(mutation)
                if mutation not in mutated_chromosome:
                    break
            mutated_chromosome[i] = mutation
    return mutated_chromosome

def roulette_selection(population, fitness):
    return random.choices(population, weights=fitness,  k=2)
    
def tournament_selection(population, fitness):
    selected_parents = []
    
    while len(selected_parents) < 2:
        tournament_candidates = random.sample(range(len(population)), 2)
        tournament_fitness = [fitness[i] for i in tournament_candidates]
        best_index = max(range(2), key=lambda x: tournament_fitness[x])
        selected_parents.append(population[tournament_candidates[best_index]])
    return selected_parents