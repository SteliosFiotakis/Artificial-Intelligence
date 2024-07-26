# 0. Problem

# 1. Encoding

# 2. Creation

# 3. Fitness

# 4. Selection

# 5. Crossover

# 6. Mutation

# 7. Solution

import random
import numpy as np

class OR1_GEN:
    def __init__(self, population, length):
        self.length = length
        self.fitness_list = list()
        self.population = population
        self.current_population = list()
        self.init_generate_population()
        self.current_fitness()
        self.generate_population()

    def init_generate_population(self):
        for _ in range(self.population):
            self.current_population.append(random.choices([i for i in range(101)], k=self.length))
        print(self.current_population)

    def generate_population(self):

        for _ in range(self.population):
            self.current_population.append(random.choices(self.current_population, weights=self.fitness_list, k=self.length))
        print(self.current_population)

    def current_fitness(self):
        self.fitness_list.clear()
        for solution in self.current_population:
            self.fitness_list.append(self.fitness_function(solution))
        print(self.fitness_list)

    @staticmethod
    def fitness_function(my_list):
        x1 = my_list[0]
        x2 = my_list[1]
        x3 = my_list[2]
        x4 = my_list[3]
        return 6 * x1 + 4 * x2 + 3 * x3 + 5 * x4

OR1_GEN(10, 4)
