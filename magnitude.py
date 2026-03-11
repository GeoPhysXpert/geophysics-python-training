magnitudes = [3.2, 4.8, 5.5, 6.9, 2.1, 7.3]
for i in magnitudes:
    if i < 4.0:
        print(i, "-> Minor")
    elif 4.0 <= i <= 5.9:
        print(i, "-> Light/Moderate")
    elif 6.0 <= i <= 6.9:
        print(i, "-> Strong")
    else:
        print(i, "-> Major")
