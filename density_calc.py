def calculate_density(mass, volume):
    try:
        return round(mass / volume, 2)
    except ZeroDivisionError:
        return "Invalid: Volume cannot be zero."
mass = 1025
volume = 1.33
print("density: ", calculate_density(mass, volume))
