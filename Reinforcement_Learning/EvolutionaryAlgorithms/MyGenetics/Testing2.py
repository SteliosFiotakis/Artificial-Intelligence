def profit(my_list):
    x1 = my_list[0]
    x2 = my_list[1]
    x3 = my_list[2]
    x4 = my_list[3]
    return 6 * x1 + 4 * x2 + 3 * x3 + 5 * x4

def broke_restrictions(my_list):
    x1 = my_list[0]
    x2 = my_list[1]
    x3 = my_list[2]
    x4 = my_list[3]
    if not (4 * x1 + 2 * x2 + 3 * x3 + x4) <= 420:
        return True
    if not (2 * x1 + 3 * x2 + x3 + 3 * x4) <= 280:
        return True
    if not (3 * x1 + x3) <= 210:
        return True
    return False

index = 3
best_solution = 0
my_variables = [0, 0, 0, 0]

while True:
    if not broke_restrictions(my_variables):
        if profit(my_variables) >= best_solution:
            best_solution = profit(my_variables)
            print(my_variables, best_solution)
        my_variables[index] += 1
    else:
        my_variables[index] = 0
        index -= 1
        if index == -1:
            break
        continue
    index = 3
