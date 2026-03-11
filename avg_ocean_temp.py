temperatures = [26.4, 27.1, 26.8, 27.5, 28.0, 26.9, 27.3]
def average_temperature(data):
    sum = 0
    for i in data:
        sum += i
    mean_naught = sum / len(data)
    mean = round(mean_naught, 3)
    return mean
print("average sea surface temperature", average_temperature(temperatures), "*C")
