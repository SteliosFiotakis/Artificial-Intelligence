#1x.82.1x5.2xx
x1 = 0
x2 = 0
x3 = 0
x4 = 0
print("1", x1, ".82.1", x2, "5.2", x3, x4)
while x1 != 9 or x2 != 9 or x3 != 5 or x4 != 5:
    x1 = x1 + 1
    if x1 == 10:
        x1 = 0
        x2 = x2 + 1
        if x2 == 10:
            x2 = 0
            x3 = x3 + 1
            if x3 == 6:
                x3 = 0
                x4 = x4 + 1
                if x4 == 10:
                    x4 = 0
    print("1", x1, ".82.1", x2, "5.2", x3, x4)