from GeneticAlgorithm import Genetic

def your_function1(my_list):
    x1 = my_list[0]
    x2 = my_list[1]
    x3 = my_list[2]
    x4 = my_list[3]
    if not (x1 >= 0 and x2 >= 0 and x3 >= 0 and x4 >= 0):
        return 0
    if not (4 * x1 + 2 * x2 + 3 * x3 + x4) <= 420:
        return 0
    if not (2 * x1 + 3 * x2 + x3 + 3 * x4) <= 280:
        return 0
    if not (3 * x1 + x3) <= 210:
        return 0
    return 6 * x1 + 4 * x2 + 3 * x3 + 5 * x4

def your_function2(my_list):
    x11 = my_list[0]
    x12 = my_list[1]
    x14 = my_list[2]
    x15 = my_list[3]
    x21 = my_list[4]
    x22 = my_list[5]
    x23 = my_list[6]
    x24 = my_list[7]
    x31 = my_list[8]
    x33 = my_list[9]
    x34 = my_list[10]
    x35 = my_list[11]
    y11 = my_list[12]
    y12 = my_list[13]
    y14 = my_list[14]
    y15 = my_list[15]
    y21 = my_list[16]
    y22 = my_list[17]
    y23 = my_list[18]
    y24 = my_list[19]
    y31 = my_list[20]
    y33 = my_list[21]
    y34 = my_list[22]
    y35 = my_list[23]

    # Total students in each district
    if x11 + x21 + x31 + y11 + y21 + y31 != 210:
        return 0
    if x12 + x22 + y12 + y22 != 240:
        return 0
    if x23 + x33 + y23 + y33 != 130:
        return 0
    if x14 + x24 + x34 + y14 + y24 + y34 != 80:
        return 0
    if x15 + x35 + y15 + y35 != 160:
        return 0

    # Extraordinary / normal students ratio for school 1 (Bus)
    if y11 / (x11 + y11) != 0.3:
        return 0
    if y12 / (x12 + y12) != 0.6:
        return 0
    if y14 / (x14 + y14) != 0.1:
        return 0
    if y15 / (x15 + y15) != 0.2:
        return 0

    # Extraordinary / normal students ratio for school 2 (Bus)
    if y21 / (x21 + y21) != 0.3:
        return 0
    if y22 / (x22 + y22) != 0.6:
        return 0
    if y23 / (x23 + y23) != 0.2:
        return 0
    if y24 / (x24 + y24) != 0.1:
        return 0

    # Extraordinary / normal students ratio for school 3 (Bus)
    if y31 / (x31 + y31) != 0.3:
        return 0
    if y33 / (x33 + y33) != 0.2:
        return 0
    if y34 / (x34 + y34) != 0.1:
        return 0
    if y35 / (x35 + y35) != 0.2:
        return 0

    # Extraordinary / normal students ratio in Schools
    students_in_school1 = x11 + x12 + x14 + x15 + y11 + y12 + y14 + y15
    students_in_school2 = x21 + x22 + x23 + x24 + y21 + y22 + y23 + y24
    students_in_school3 = x31 + x33 + x34 + x35 + y31 + y33 + y34 + y35
    extraordinary_in_school1 = y11 + y12 + y14 + y15
    extraordinary_in_school2 = y21 + y22 + y23 + y24
    extraordinary_in_school3 = y31 + y33 + y34 + y35

    if not (students_in_school1 / 5 <= extraordinary_in_school1 <= students_in_school1 / 2):
        return 0
    if not (students_in_school2 / 5 <= extraordinary_in_school2 <= students_in_school2 / 2):
        return 0
    if not (students_in_school3 / 5 <= extraordinary_in_school3 <= students_in_school3 / 2):
        return 0

    # Total students in schools
    if students_in_school1 > 400:
        return 0
    if students_in_school2 > 300:
        return 0
    if students_in_school3 > 200:
        return 0

    # FINALLY COMBINATION IS VALID!!!
    return 0.6 * (x11 + y11) + 0.2 * (x12 + y12) + 0.7 * (x14 + y14) + 0.7 * (x15 + y15)\
        + 0.4 * (x21 + y21) + x22 + y22 + 0.25 * (x23 + y23) + 0.35 * (x24 + y24)\
        + 0.65 * (x31 + y31) + 0.8 * (x33 + y33) + x34 + y34 + x35 + y35

def my_function(my_list):
    x = my_list[0]
    y = my_list[1]
    return x ** 3 + 15 * x ** 2 + y ** 3 + 15 * y ** 2

population_length = 1_000
individual_length = 4
mutation_factor = 5
mutation_decay = 1_000
mutation_rate = 0.1
max_value = 100

algorithm = Genetic(population_length, individual_length, max_value, mutation_rate, mutation_factor, mutation_decay,
                    your_function1)

algorithm.start()
