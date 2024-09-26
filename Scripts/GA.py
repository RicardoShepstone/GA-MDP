# Clases con los algoritmos geneticos
# autor: Ricardo Ignacio Shepstone Aramburu

import math
import numpy as np
import random


class MDP_int:
    """
    Algorimo genético para el problema MDP con representación de enteros
    
    """
    
    # Definimos el constructor con los atributos especificados.
    def __init__(self, population_size, pm, generations=200, tournament_size = False):
        self.__population_size = population_size
        self.__pm = pm
        self.__generations = generations
        self.__tournament_size = tournament_size
    
    # Lee 
    def __read_distances(self, file_name: str):
        distances = {}
        with open(file_name,'r') as file:
            self.__n,self.__m = map(int, file.readline().split())
            for line in file:
                element1, element2, distance = map(float, line.split())
                element1, element2 = int(element1), int(element2)
                distances[(element1,element2)] = distance
        return distances
    
    # Genera una solucion con la representacion de enteros
    def __generate_chromosome(self):
        return list(np.random.choice(self.__n, size = self.__m, replace = False))
    
    # Genera una poblacion de soluciones
    def __generate_population(self):
        return [self.__generate_chromosome() for p in range(self.__population_size)]
    
    
    # Funcion para el calculo de la funcion objetivo para una solucion
    def __calculate_fitness(self, chromosome, distances):

        dist_list = []
        for i in chromosome:
            for j in chromosome[chromosome.index(i)+1:]:
                if j>i:
                    dist_list.append(distances[(i,j)])
                else:
                    dist_list.append(distances[(j,i)])
                    
        return math.fsum(dist_list)
    
    """ La implementada es mas eficiente
    
        dist_list = []
        for i in chromosome:
            for j in chromosome:
                if j>i:
                    dist_list.append(distances[(i,j)])
                elif i>j:
                    dist_list.append(distances[(j,i)])
                    
        return math.fsum(dist_list)/2
    """
    
    
    # Funcion para evaluar a la poblacion entera
    def __evaluate_population(self, population, distances):
        return [self.__calculate_fitness(chromosome, distances) for chromosome in population]
        
    
    # Operador de cruce
    def __crossover_operation(self, parent1, parent2):
        common_elements = list(set(parent1).intersection(set(parent2)))
        parent1_copy = parent1.copy()
        parent2_copy = parent2.copy()
        for element in common_elements:
            parent1_copy.remove(element)
            parent2_copy.remove(element)
        crossover_point = random.randint(1,len(parent1)-1)
        offspring1 = parent1_copy[0:crossover_point]+common_elements+parent2_copy[crossover_point:]
        offspring2 = parent2_copy[0:crossover_point]+common_elements+parent1_copy[crossover_point:]
        return offspring1, offspring2
    
    # Operador de mutacion
    def __mutation_operation(self, chromosome):
        mutated_chromosome = chromosome.copy()
        for i in range(len(chromosome)):
            if random.random()<self.__pm:
                while True:
                    mutation = random.randint(0,self.__n-1)
                    if mutation not in mutated_chromosome:
                        break
                mutated_chromosome[i] = mutation
        return mutated_chromosome
        
    # Seleccion por ruleta
    def __roulette_selection(self, population, fitness):
        return random.choices(population, weights=fitness,  k=2)
    
   
    # Seleccion por torneo
    def __tournament_selection(self, population, fitness):
        winner_parents = []
        while len(winner_parents)<2:
            candidates_index = random.sample(range(self.__population_size), self.__tournament_size)
            winner_index = min(candidates_index)
            winner_parents.append(population[winner_index])
        return winner_parents
            
        """
        Si las listas population y fitness no estuviesen ordenadas
        
        def __tournament_selection(self, population, fitness):
            selected_parents = []
            
            while len(selected_parents) < 2:
                tournament_candidates = random.sample(range(self.__population_size), self.__tournament_size)
                tournament_fitness = [fitness[i] for i in tournament_candidates]
                best_index = max(range(tournament_size), key=lambda x: tournament_fitness[x])
                selected_parents.append(population[tournament_candidates[best_index]])
            return selected_parents
            
        """
        
    
    def run_evol(self, file_name):
        distance_dict = self.__read_distances(file_name)
        current_population = self.__generate_population()
        #fitness_evolution = np.array([])
        for i in range(self.__generations):
            print('generation: '+str(i))
            current_population = sorted(
                current_population, 
                key = lambda chromosome: self.__calculate_fitness(chromosome, distance_dict), 
                reverse=True
                )
            population_fitness = sorted(self.__evaluate_population(current_population, distance_dict), reverse=True)
            #fitness_evolution = np.append(fitness_evolution, np.array(population_fitness).max()) 
            new_population = []
            new_population += current_population[:2]
            
            while len(new_population)<self.__population_size:
                if self.__tournament_size:
                    parents = self.__tournament_selection(current_population, population_fitness)
                else:
                    parents = self.__roulette_selection(current_population, population_fitness)
                
                if parents[0]!=parents[1]:
                    child1, child2 = self.__crossover_operation(parents[0], parents[1])
                    child1 = self.__mutation_operation(child1)
                    child2 = self.__mutation_operation(child2)
                    new_population += [child1, child2]
            
            current_population = new_population
        
        return current_population, self.__evaluate_population(current_population, distance_dict)#, fitness_evolution
                
