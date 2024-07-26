from Population import Population

population = None

def setup():
    global population
    target = "Hello World"
    population_n = 200
    mutation_rate = 0.01

    population = Population(target, mutation_rate, population_n)

def draw():
    global population
    population.natural_selection()
    population.generate()
    population.calculate_fitness()

    population.evaluate()

    if population.is_finished():
        return True

    display()

def display():
    global population
    answer = population.get_best()

    print(f"Best phrase: {answer}")
    # print(f"Total generations: {population.get_generations()}")
    # print(f"All phrases: \n{population.all_phrases()}")

setup()
while True:
    if draw():
        break

print(population.get_best())
