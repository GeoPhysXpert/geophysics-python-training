import math
a_naught = 10
k = 0.05
x = 100
a_final = a_naught * math.exp(-k * x)
a_f = round(a_final, 3)
print("the final amplitude is =", a_f)
