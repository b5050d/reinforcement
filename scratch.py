from matplotlib import pyplot as plt



hourly_rate = 60
drive_time = 1 - .55
days_in_office = 3
weekly_driving = drive_time * days_in_office * 2
print(f"Weekly driving per week: {weekly_driving}")

driving_cost = (weekly_driving * hourly_rate) * (52*(9/12))
print(f"Cost of driving per week: {weekly_driving}")

rental_increase = 12000

increased_cost = rental_increase + driving_cost

print(increased_cost)


# What would my stock be worth if we get series A funding

# ill take 100k in pay but 40k in stock options

"""

Stock options - how does this work before there is a stock

You will be granted something in terms of stock options

# Lets say hes confident that the stock is going to be worth
1000 dollars per share


Usually 2 or 3 multiple

100k and 40 shares (40 shares could end up to be 120)
"""