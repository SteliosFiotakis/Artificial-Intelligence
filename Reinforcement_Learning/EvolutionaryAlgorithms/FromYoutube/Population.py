import random

from DNA import DNA

class Population:
    def __init__(self, target, mutation_rate, population_n):
        self.population = [i for i in range(population_n)]
        self.mating_pool = list()
        self.generations = 0
        self.finished = False
        self.target = target
        self.mutation_rate = mutation_rate
        self.perfect_score = 1

        self.best = ""

        for i in range(population_n):
            self.population[i] = DNA(len(self.target))

        self.calculate_fitness()

    def calculate_fitness(self):
        for i in range(len(self.population)):
            self.population[i].calculate_fitness(self.target)

    def natural_selection(self):
        self.mating_pool = list()

        max_fitness = 0
        for i in range(len(self.population)):
            if self.population[i].fitness > max_fitness:
                max_fitness = self.population[i].fitness

        if max_fitness == 0:
            for i in range(len(self.population)):
                self.mating_pool.append(self.population[i])
        else:
            for i in range(len(self.population)):
                fitness = self.population[i].fitness / max_fitness
                n = int(fitness * 100)
                for j in range(n):
                    self.mating_pool.append(self.population[i])

    def generate(self):
        for i in range(len(self.population)):
            a = random.randint(0, len(self.mating_pool)-1)
            b = random.randint(0, len(self.mating_pool)-1)
            partner_a = self.mating_pool[a]
            partner_b = self.mating_pool[b]
            child = partner_a.crossover(partner_b)
            child.mutate(self.mutation_rate)
            self.population[i] = child

        self.generations += 1

    def get_best(self):
        return self.best

    def evaluate(self):
        world_record = 0.0
        index = 0
        for i in range(len(self.population)):
            if self.population[i].fitness > world_record:
                index += 1
                world_record = self.population[i].fitness

        self.best = self.population[index].get_phrase()
        if world_record == self.perfect_score:
            self.finished = True

    def is_finished(self):
        return self.finished

    def get_generations(self):
        return self.generations

    def get_average_fitness(self):
        total = 0
        for i in range(len(self.population)):
            total += self.population[i].fitness

        return total / len(self.population)

    def all_phrases(self):
        everything = str()
        display_limit = min(len(self.population), 50)

        for i in range(display_limit):
            everything += self.population[i].get_phrase() + "\n"

        return everything
