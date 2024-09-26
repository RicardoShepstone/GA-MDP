# -*- coding: utf-8 -*-
"""
Created on Sun May 28 23:42:08 2023

@author: richa
"""
import Funciones as F
import random
import numpy as np
from GA import MDP_int
import sklearn
from sklearn.model_selection import ParameterGrid
import time

# Pruebas lectura
ruta = "Instancias y Tablas MDP 2020-21\MDG-a_1_n500_m50.txt"
distances = F.read_distances(ruta)

# Pruebas del cálculo de la función objetivo
ex_chromosome = [0,7,8,20]
value = distances[(0,7)]+distances[(0,8)]+distances[(0,20)]+distances[(7,8)]+distances[(7,20)]+distances[(8,20)] 

value2 = F.calculate_fitness(ex_chromosome, distances)

ex_chromosome2 = [0,20,8,7]
value3 = F.calculate_fitness(ex_chromosome2, distances)

# Pruebas para generar cromosomas
chromosome1 = F.generate_chromosome(50,500)
chromosome2 = F.generate_chromosome(50,500)

# Pruebas para generar población
population1 = F.generate_population(20)
F.calculate_fitness(population1[0], distances)

# Pruebas para el operador de cruce
common_elements = set(chromosome1).intersection(set(chromosome2))

for element in common_elements:
    chromosome1.remove(element)
    chromosome2.remove(element)

child1, child2 = F.crossover_operation(chromosome1.copy(), chromosome2.copy())

F.calculate_fitness(chromosome1, distances)
F.calculate_fitness(chromosome2, distances)
F.calculate_fitness(child1, distances)
F.calculate_fitness(child2, distances)

# Pruebas para el operador de mutación
mutated_child1 = F.mutation_operation(child1, 0.01)
print(mutated_child1 == child1)
F.calculate_fitness(child1, distances)
F.calculate_fitness(mutated_child1, distances)

# Calcular la función objetivo para la población entera
population2 = F.generate_population(5)
F.evaluate_population(population1, distances)
fitness = F.evaluate_population(population2, distances)
fitness2 = [F.calculate_fitness(chromosome, distances) for chromosome in population2]
print(fitness == fitness2)

# Pruebas para ordenar la poblacion segun la funcion objetivo
sorted(fitness2, reverse=True)[:2]
population2_sorted = sorted(population2, key = lambda chromosome: F.calculate_fitness(chromosome, distances),reverse=True)
fitness_sorted = sorted(fitness, reverse=True)

# Pruebas para la seleccion
random.choices(population2, weights=fitness,  k=2)[0]

# Pruebas para la selección de torneo
"""
tournament_candidates = random.sample(range(len(population2)), 2)
tournament_fitness = [fitness[i] for i in tournament_candidates]
best_index = max(range(2), key=lambda x: tournament_fitness[x])
selected_parents = population2[tournament_candidates[best_index]]
"""
print(F.tournament_selection(population2, fitness))

candidates_index = random.sample(range(len(population2_sorted)), 2)
winner_index = min(candidates_index)
selected_parent = population2_sorted[winner_index]

# Experimentos
ga = MDP_int(5000, 0.01)
pop, fit, evolution = ga.run_evol(ruta)

ga2 = MDP_int(500, 0.05)
pop2, fit2, evolution2 = ga2.run_evol(ruta)

ga3 = MDP_int(500, 0.1)
pop3, fit3, evolution3 = ga3.run_evol(ruta)

ga4 = MDP_int(1000, 0.2)
pop4, fit4, evolution4 = ga4.run_evol(ruta)

ga5 = MDP_int(500, 0.1)
pop5, fit5, evolution5 = ga5.run_evol(ruta)

ga5 = MDP_int(500, 0.5)
pop5, fit5, evolution5 = ga5.run_evol(ruta)

ga6 = MDP_int(1000, 0.1)
pop6, fit6, evolution6 = ga6.run_evol(ruta)

ga7 = MDP_int(1000, 0.1)
pop7, fit7, evolution7 = ga7.run_evol(ruta)

# Probamos con un conjunto mayor
ruta2 = "Instancias y Tablas MDP 2020-21\MDG-c_20_n3000_m600.txt"
ga_c = MDP_int(500, 0.01)
pop_c, fit_c, evolution_c = ga_c.run_evol(ruta2)



# Pruebas para determinar si los experimentos son reproducibles
ruta3 = "Instancias y Tablas MDP 2020-21\MDG-a_4_n500_m50.txt"
np.random.seed(1234)
random.seed(1234)
ga_test1 = MDP_int(100, 0.01, 50)
best_fitness_test1 = []
for i in range(10):
    pop_test1, fit_test1 = ga_test1.run_evol(ruta3)
    best_fitness_test1.append(np.max(fit_test1))

np.random.seed(1234)
random.seed(1234)
ga_test2 = MDP_int(200, 0.01, 50)
best_fitness_test2 = []
for i in range(10):
    pop_test2, fit_test2 = ga_test2.run_evol(ruta3)
    best_fitness_test2.append(np.max(fit_test2))
    
# pruebas con seleccion por torneo
np.random.seed(1234)
random.seed(1234)
ga2_sel = MDP_int(200, 0.01, tournament_size=2)
pop2_sel, fit2_sel = ga2_sel.run_evol(ruta2)



# Pruebas para determinar los mejores parámetros
ruta4 = "Instancias y Tablas MDP 2020-21\MDG-a_8_n500_m50.txt"

param_grid = {'population_size': [200,400,600], 'pm' : [0.01, 0.05, 0.1], 'iterations': [50, 100, 150]}
grid = ParameterGrid(param_grid)

np.random.seed(1234)
random.seed(1234)

results_dict = {}
params_dict = {}
p = 0
for params in grid: 
    print(str(p)+' de '+str(len(grid)))
    ga_param = MDP_int(params['population_size'], params['pm'], params['iterations'])
    fitness_array = np.array([])
    time_array = np.array([])
    for i in range(5):
        t0 = time.time()
        pop_param, fit_param = ga_param.run_evol(ruta4)
        time_array = np.append(time_array, time.time()-t0)
        fitness_array = np.append(fitness_array, np.max(fit_param))
        
    results_dict[p] = [np.average(fitness_array),np.std(fitness_array),np.average(time_array),np.std(time_array)]
    params_dict[p] = params
    p += 1
    
    
""" Mejores scores:
    [6823.140000000001, 41.43510733665355, 60.02365312576294, 5.294623060878562]
    {'iterations': 150, 'pm': 0.01, 'population_size': 400}
    
    [6859.94, 43.23975346830725, 29.477833223342895, 0.734081666858303]
    {'iterations': 150, 'pm': 0.01, 'population_size': 200}
    
    [6771.557999999999, 31.66092253867527, 35.19368748664856, 0.6478449634621359]
    {'iterations': 100, 'pm': 0.01, 'population_size': 400}
    
    [6764.066000000001, 40.076602450806824, 16.025563955307007, 0.26651943138563017]
    {'iterations': 100, 'pm': 0.01, 'population_size': 200}
"""


ruta4 = "Instancias y Tablas MDP 2020-21\MDG-a_8_n500_m50.txt"


np.random.seed(1234)
random.seed(1234)

results_dict_sel = {}
params_dict_sel = {}
p = 0
for params in grid: 
    print(str(p)+' de '+str(len(grid)))
    ga_param_sel = MDP_int(params['population_size'], params['pm'], params['iterations'], tournament_size=2)
    fitness_array_sel = np.array([])
    time_array_sel = np.array([])
    for i in range(5):
        t0 = time.time()
        pop_param_sel, fit_param_sel = ga_param_sel.run_evol(ruta4)
        time_array_sel = np.append(time_array_sel, time.time()-t0)
        fitness_array_sel = np.append(fitness_array_sel, np.max(fit_param_sel))
        
    results_dict_sel[p] = [np.average(fitness_array_sel),np.std(fitness_array_sel),np.average(time_array_sel),np.std(time_array_sel)]
    params_dict_sel[p] = params
    p += 1
    
    
""" Mejores scores:
    [7449.7699999999995, 42.378872566409676, 61.50932126045227, 0.7704242271844469]
    {'iterations': 150, 'pm': 0.01, 'population_size': 600}
    
    [7448.389999999999, 72.80637554500302, 39.97456064224243, 0.16138863543772025]
    {'iterations': 150, 'pm': 0.01, 'population_size': 400}
    
    [7376.759999999999, 29.998641969262618, 19.683125448226928, 0.23636614109628062]
    {'iterations': 150, 'pm': 0.01, 'population_size': 200}
    
    [7361.817999999999, 26.9285895657384, 41.98205695152283, 0.4865060828050417]
    {'iterations': 100, 'pm': 0.01, 'population_size': 600}
"""
    