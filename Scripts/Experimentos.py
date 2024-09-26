# Script para los experimentos
# autor: Ricardo Ignacio Shepstone Aramburu

import os
import pandas as pd
import numpy as np
import random
from GA import MDP_int
import time

directory = './Instancias y Tablas MDP 2020-21/'  # Ruta al directorio que contiene los archivos de texto

# Obtener la lista de nombres de archivos en el directorio
file_names = os.listdir(directory)

# Declaramos un DataFrame para guardar los resultados
resultados = pd.DataFrame(columns=['Caso', 'Coste', 'tiempo'])

# Declaramos diccionario para guardar las poblaciones y coste
population_dict = {}
fitness_dict = {}

# Fijamos semillas
np.random.seed(1234)
random.seed(1234)

# Declaramos el modelo
ga_MDP = MDP_int(200, 0.01, 150, tournament_size=2)

# Leer cada archivo individualmente y realizamos experimento
for file_name in file_names:
    file_path = os.path.join(directory, file_name)  # Ruta completa al archivo
    if file_path.endswith('.txt'):
        t0 = time.time()
        population, fitness = ga_MDP.run_evol(file_path)
        delta_time = time.time()-t0
    
    # Guardamos resultados
    population_dict[file_name] = population
    fitness_dict[file_name] = fitness
    nueva_fila = pd.DataFrame({'Caso':[file_name],'Coste':[np.max(fitness)],'tiempo':[delta_time]})
    resultados = pd.concat([resultados, nueva_fila], ignore_index=True)