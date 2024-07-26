import random

class Genetic:
    def __init__(self, population_length, individual_length, max_value, mutation_rate, mutation_factor, mutation_decay,
                 the_fitness_function):
        self.fitness_function = the_fitness_function
        self.population_length = population_length
        self.individual_length = individual_length
        self.mutation_factor = mutation_factor
        self.mutation_decay = mutation_decay
        self.mutation_rate = mutation_rate
        self.current_population = list()
        self.previous_best_solution = 0
        self.best_combination = None
        self.selection_list = list()
        self.max_value = max_value
        self.best_solution = 0
        self.parent_1 = None
        self.parent_2 = None
        self.cnt = 0

    def fitness_calculation(self, my_list):
        return self.fitness_function(my_list)

    def create_initial_population(self):
        for _ in range(self.population_length):
            individual = [random.randint(-self.max_value, self.max_value) for _ in range(self.individual_length)]
            self.current_population.append(individual)

    def select(self):
        self.selection_list = list()
        fitness_indexer = [self.fitness_calculation(elem) for elem in self.current_population]
        max_fitness = max(fitness_indexer)

        if max_fitness == 0:
            for individual in self.current_population:
                self.selection_list.append(individual)
        else:
            if max_fitness > self.best_solution:
                self.best_solution = max_fitness
                self.best_combination = self.current_population[fitness_indexer.index(max_fitness)]
            for index, fitness_value in enumerate(fitness_indexer):
                int_percentage = int(fitness_value / max_fitness * 100)
                for _ in range(int_percentage):
                    self.selection_list.append(self.current_population[index])

        self.parent_1 = random.choice(self.selection_list)
        self.parent_2 = random.choice(self.selection_list)

    def crossover(self):
        child = list()
        midpoint = random.randint(0, self.individual_length-1)

        for i in range(self.individual_length):
            if i > midpoint:
                child.append(self.parent_1[i])
            else:
                child.append(self.parent_2[i])

        return child

    def mutate(self):
        for individual in range(self.population_length):
            for index in range(self.individual_length):
                if random.random() < self.mutation_rate:
                    self.current_population[individual][index] += \
                        random.randint(-self.mutation_factor, self.mutation_factor)

    def start(self):
        self.create_initial_population()
        while True:
            self.select()
            self.current_population.clear()
            for _ in range(self.population_length):
                self.current_population.append(self.crossover())
            self.mutate()
            if self.best_solution > self.previous_best_solution:
                self.previous_best_solution = self.best_solution
                print(self.best_solution, self.best_combination)
            # if self.cnt % self.mutation_decay == 0:
            #     if self.mutation_factor == 0:
            #         break
            #     self.mutation_factor -= 1
            self.cnt += 1
