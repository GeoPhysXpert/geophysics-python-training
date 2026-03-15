def calculate_distance(a, b):

    if a > b:
        distance = a - b
    else:
        distance = b - a
    return distance
print("distance between stations =", calculate_distance(310, 125), "km")
