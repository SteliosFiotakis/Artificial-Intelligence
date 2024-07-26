calendar_a = [['9:00', '10:30'], ['12:00', '13:00'], ['16:00', '18:00']]
calendar_a_bounds = ['9:00', '20:00']

calendar_b = [['10:00', '11:30'], ['12:30', '14:30'], ['14:30', '15:00'], ['16:00', '17:00']]
calendar_b_bounds = ['10:00', '18:30']

meeting_time = 30


def find_time_for_meeting(a, abounds, b, bbounds, time):
    for i in range(len(a)):
        a[i][0] = a[i][0].replace(':', '')
        a[i][1] = a[i][1].replace(':', '')

    for i in range(len(b)):
        b[i][0] = b[i][0].replace(':', '')
        b[i][1] = b[i][1].replace(':', '')

    for i in range(2):
        abounds[i] = abounds[i].replace(':', '')
        bbounds[i] = abounds[i].replace(':', '')

    limits = ['', '']
    limits[0] = abounds[0] if abounds[0] > bbounds[0] else bbounds[0]
    limits[1] = abounds[1] if abounds[1] < bbounds[1] else bbounds[1]
    return limits


print(find_time_for_meeting(calendar_a, calendar_a_bounds, calendar_b, calendar_b_bounds, meeting_time))
