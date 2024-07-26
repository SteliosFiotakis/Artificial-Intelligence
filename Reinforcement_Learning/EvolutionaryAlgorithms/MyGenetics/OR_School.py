import numpy as np

def get_district_combinations2(people):
    temp_list = list()
    for a in range(people+1):
        for b in range(people+1):
            if a + b == people:
                temp_list.append((a, b))
    return np.asarray(temp_list)

def get_district_combinations3(people):
    temp_list = list()
    for a in range(people+1):
        for b in range(people+1):
            for c in range(people+1):
                if a + b + c == people:
                    temp_list.append((a, b, c))
    return np.asarray(temp_list)

def get_correct_percentages2(x_array, y_array, percentage):
    temp_x = list()
    temp_y = list()
    for x in x_array:
        if x[0] and x[1]:
            for y in y_array:
                if y[0] and y[1] and ((y[0] / (x[0] + y[0])) == percentage) and ((y[1] / (x[1] + y[1])) == percentage):
                    temp_x.append(x)
                    temp_y.append(y)
    return np.asarray(temp_x), np.asarray(temp_y)

def get_correct_percentages3(x_array, y_array, percentage):
    temp_x = list()
    temp_y = list()
    for x in x_array:
        if x[0] and x[1]:
            for y in y_array:
                if y[0] and y[1] and y[2] and ((y[0] / (x[0] + y[0])) == percentage) and \
                        ((y[1] / (x[1] + y[1])) == percentage) and ((y[2] / (x[2] + y[2])) == percentage):
                    temp_x.append(x)
                    temp_y.append(y)
    return np.asarray(temp_x), np.asarray(temp_y)

def districts_to_schools(x_district_1, x_district_2, x_district_3, x_district_4,
                         y_district_1, y_district_2, y_district_3, y_district_4,
                         total_students):
    temp_x_school = list()
    temp_y_school = list()
    for a in np.unique(x_district_1):
        for b in np.unique(x_district_2):
            for c in np.unique(x_district_3):
                for d in np.unique(x_district_4):
                    for e in np.unique(y_district_1):
                        for f in np.unique(y_district_2):
                            for g in np.unique(y_district_3):
                                for h in np.unique(y_district_4):
                                    xs = a + b + c + d
                                    ys = e + f + g + h
                                    # print(a, b, c, d, e, f, g, h)
                                    # print((xs + ys) <= total_students)
                                    # print(((xs + ys) / 5), (ys / (xs + ys)), ((xs + ys) / 2))

                                    if ((xs + ys) <= total_students) and (0.2 <= (ys / (xs + ys)) <= 0.5):
                                        # print('hi')
                                        print(a, b, c, d, e, f, g, h)
                                        temp_x_school.append((a, b, c, d))
                                        temp_y_school.append((e, f, g, h))
    return np.asarray(temp_x_school), np.asarray(temp_y_school)

x_district1 = get_district_combinations3(147)
x_district2 = get_district_combinations2(96)
x_district3 = get_district_combinations2(104)
x_district4 = get_district_combinations3(72)
x_district5 = get_district_combinations2(128)

y_district1 = get_district_combinations3(63)
y_district2 = get_district_combinations2(144)
y_district3 = get_district_combinations2(26)
y_district4 = get_district_combinations3(8)
y_district5 = get_district_combinations2(32)

def print_values():
    print(x_district1, len(x_district1))
    print()
    print(x_district2, len(x_district2))
    print()
    print(x_district3, len(x_district3))
    print()
    print(x_district4, len(x_district4))
    print()
    print(x_district5, len(x_district5))
    print()

    print(y_district1, len(y_district1))
    print()
    print(y_district2, len(y_district2))
    print()
    print(y_district3, len(y_district3))
    print()
    print(y_district4, len(y_district4))
    print()
    print(y_district5, len(y_district5))
    print()
    print()
    print()
    print()
    print()

print_values()

x_district1, y_district1 = get_correct_percentages3(x_district1, y_district1, 0.3)
x_district2, y_district2 = get_correct_percentages2(x_district2, y_district2, 0.6)
x_district3, y_district3 = get_correct_percentages2(x_district3, y_district3, 0.2)
x_district4, y_district4 = get_correct_percentages3(x_district4, y_district4, 0.1)
x_district5, y_district5 = get_correct_percentages2(x_district5, y_district5, 0.2)

print_values()

x_school3, y_school3 = districts_to_schools(x_district1[:, 2], x_district3[:, 1], x_district4[:, 2], x_district5[:, 1],
                                            y_district1[:, 2], y_district3[:, 1], y_district4[:, 2], y_district5[:, 1],
                                            200)

# print(x_district1[:, 2])
# print()
# print(x_district3[:, 1])
# print()
# print(x_district4[:, 2])
# print()
# print(x_district5[:, 1])
# print()
# print()
# print()
# print(y_district1[:, 2])
# print()
# print(y_district3[:, 1])
# print()
# print(y_district4[:, 2])
# print()
# print(y_district5[:, 1])
