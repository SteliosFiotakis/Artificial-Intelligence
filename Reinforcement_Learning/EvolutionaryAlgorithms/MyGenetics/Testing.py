def profit(x1, x2, x3, x4):
    return 6 * x1 + 4 * x2 + 3 * x3 + 5 * x4

def broke_restrictions(x1, x2, x3, x4):
    if not (4 * x1 + 2 * x2 + 3 * x3 + x4) <= 420:
        return True
    if not (2 * x1 + 3 * x2 + x3 + 3 * x4) <= 280:
        return True
    if not (3 * x1 + x3) <= 210:
        return True
    return False

best_solution = 0

for one in range(500):
    for two in range(500):
        for three in range(500):
            for four in range(500):
                if broke_restrictions(one, two, three, four):
                    break
                if profit(one, two, three, four) >= best_solution:
                    best_solution = profit(one, two, three, four)
                    print(f'x1:{one}, x2:{two}, x3:{three}, x4:{four}, profit:{best_solution}')
