import random

class DNA:
    def __init__(self, lenght):
        self.genes = [i for i in range(lenght)]
        self.fitness = 0
        for i in range(lenght):
            self.genes[i] = self.new_char()

    @staticmethod
    def new_char():
        character = random.randint(63, 122)
        if character == 63:
            character = 32
        elif character == 64:
            character = 46
        return chr(character)

    def get_phrase(self):
        return "".join(self.genes)

    def calculate_fitness(self, target):
        score = 0
        for i in range(len(self.genes)):
            if self.genes[i] == target[i]:
                score += 1
        self.fitness = score / len(target)

    def crossover(self, partner):
        child = DNA(len(self.genes))
        midpoint = random.randint(0, len(self.genes))

        for i in range(len(self.genes)):
            if i > midpoint:
                child.genes[i] = self.genes[i]
            else:
                child.genes[i] = partner.genes[i]
        return child

    def mutate(self, mutation_rate):
        for i in range(len(self.genes)):
            if random.random() < mutation_rate:
                self.genes[i] = self.new_char()
