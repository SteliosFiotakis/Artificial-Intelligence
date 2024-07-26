while True:
    Size = int(input('Size: '))
    Sum = int(input('Sum: '))
    my_list = list()
    p_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    for elem1 in p_numbers:
        temp_list = Size * [0]
        temp_list[0] = elem1
        list1 = p_numbers.copy()
        list1.remove(elem1)
        if Size == 1:
            my_list.append(temp_list)
        if Size > 1:
            for elem2 in list1:
                temp_list[1] = elem2
                list2 = list1.copy()
                list2.remove(elem2)
                if Size == 2:
                    my_list.append(temp_list.copy())
                if Size > 2:
                    for elem3 in list2:
                        temp_list[2] = elem3
                        list3 = list2.copy()
                        list3.remove(elem3)
                        if Size == 3:
                            my_list.append(temp_list.copy())
                        if Size > 3:
                            for elem4 in list3:
                                temp_list[3] = elem4
                                print(temp_list)
                                list4 = list3.copy()
                                list4.remove(elem4)
                                if Size == 4:
                                    my_list.append(temp_list.copy())
                                if Size > 4:
                                    for elem5 in list4:
                                        temp_list[4] = elem5
                                        list5 = list4.copy()
                                        list5.remove(elem5)
                                        if Size == 5:
                                            my_list.append(temp_list.copy())
                                        if Size > 5:
                                            for elem6 in list5:
                                                temp_list[5] = elem6
                                                list6 = list5.copy()
                                                list6.remove(elem6)
                                                if Size == 6:
                                                    my_list.append(temp_list.copy())
                                                if Size > 6:
                                                    for elem7 in list6:
                                                        temp_list[6] = elem7
                                                        list7 = list6.copy()
                                                        list7.remove(elem7)
                                                        if Size == 7:
                                                            my_list.append(temp_list.copy())
                                                        if Size > 7:
                                                            for elem8 in list7:
                                                                temp_list[7] = elem8
                                                                list8 = list7.copy()
                                                                list8.remove(elem8)
                                                                if Size == 8:
                                                                    my_list.append(temp_list.copy())
                                                                if Size > 8:
                                                                    for elem9 in list8:
                                                                        temp_list[8] = elem9
                                                                        list9 = list8.copy()
                                                                        list9.remove(elem9)
                                                                        my_list.append(temp_list)

    new_list = my_list.copy()
    for element in my_list:
        if sum(element) != Sum:
            new_list.remove(element)

    my_list = new_list.copy()
    my_set = set()
    for element in my_list:
        my_set.add(frozenset(element))

    for element in my_set:
        print(tuple(sorted(list(element))))
    print()
