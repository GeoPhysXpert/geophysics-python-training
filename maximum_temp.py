temperatures = [26.5, 27.1, 25.9, 28.3, 27.8, 26.9]
temp_max = 26.5
temp_ind = 0
for i in range(len(temperatures)):
    if temperatures[i] >= temp_max:
        temp_max = temperatures[i]
        temp_ind = i
print("maximum temperature = ", temp_max, "at index", temp_ind)
