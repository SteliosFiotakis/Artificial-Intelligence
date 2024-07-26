principal = 100
interest_rate = 5
years = 7

value = principal * (1 + (interest_rate / 100)) ** years
print(value)
print()


for year in range(years + 1):
    value = principal * (1 + (interest_rate / 100)) ** year
    print(year, value)


print()
principal = float(input('Give principal: '))
interest_rate = float(input('Give interest rate: '))
years = int(input('Give years: '))

value = principal * (1 + (interest_rate / 100)) ** years
print(value)


def find_equation(fprincipal, frate, fyears):
    temp_dict = dict()
    for fyear in range(fyears + 1):
        fvalue = fprincipal * (1 + (frate / 100)) ** fyear
        temp_dict[fyear] = fvalue
    return temp_dict


data = find_equation(principal, interest_rate, years)

print()
for key, value in data.items():
    print(key, value)
