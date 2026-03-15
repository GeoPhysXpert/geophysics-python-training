#Mini-Project 01 by Anurag Kumar
#Title - Estimating Earthquake Distance using P-S Arrival Times
#This dataset has been AI generated for the sake of this project work only!


import numpy as np
import matplotlib.pyplot as plt
#p-wave arrival times (seconds)
p_arrival = [
8.4, 9.1, 10.2, 7.8, 11.5, 12.3, 9.7, 8.9, 10.8, 11.1,
9.5, 8.2, 10.4, 11.8, 12.6, 7.9, 9.3, 10.1, 11.4, 12.0,
8.7, 9.9, 10.6, 11.2, 12.4, 7.6, 8.8, 9.4, 10.3, 11.7,
12.8, 8.1, 9.0, 10.5, 11.6, 12.2, 7.7, 8.5, 9.6, 10.7,
11.3, 12.7, 8.0, 9.2, 10.0, 11.9, 12.1, 8.3, 9.8, 10.9
]

#s-wave arrival times (seconds)
s_arrival = [
15.7, 16.9, 18.4, 14.6, 20.3, 21.1, 17.5, 16.2, 19.0, 19.7,
17.1, 15.2, 18.8, 20.6, 21.9, 14.9, 16.7, 18.1, 20.0, 20.9,
15.9, 17.8, 18.9, 19.8, 21.4, 14.3, 16.0, 17.2, 18.3, 20.5,
22.1, 15.0, 16.4, 18.7, 20.2, 21.0, 14.5, 15.6, 17.6, 18.6,
19.4, 21.8, 15.1, 16.6, 17.9, 20.8, 21.2, 15.4, 17.3, 19.2
]
p = np.array(p_arrival)
s = np.array(s_arrival)
if len(s) != len(p): #ensuring equal length of both arrays of arrival times
    raise ValueError("error! length mismatch of arrays.")
if np.any(s <= p):
    print("some s-waves reach before p-waves! physics error.")

#p-s time difference
ps_difference = s - p

vp = 6.0 #assumed crustal p-wave velocity in km/s
vs = 3.5 #assumed crustal s-wave velocity in km/s

#computing the earthquake distance
distance = ps_difference / ((1 / vs) - (1 / vp)) #using  Formula: D = (Ts - Tp) / (1/Vs - 1/Vp)


#results table
print("\nEarthquake Distance Estimates")
print("-" * 60)
print(f"{'Event':<6}{'P(s)':<10}{'S(s)':<10}{'P-S(s)':<10}{'Distance(km)':<15}") 
"""
f-string usage where each number indicates. . .
. . .the width of the column in characters.
"""
print("-" * 60)

for i in range(len(p)):
    print(f"{i+1:<6}{p[i]:<10.2f}{s[i]:<10.2f}{ps_difference[i]:<10.2f}{distance[i]:<15.2f}")


mean_distance = np.mean(distance)
max_distance = np.max(distance)
min_distance = np.min(distance)
std_distance = np.std(distance)

print(f"Mean Distance: {mean_distance:.2f} km")
print(f"Maximum Distance: {max_distance:.2f} km")
print(f"Minimum Distance: {min_distance:.2f} km")
print(f"Standard Deviation: {std_distance:.2f} km")

#classifying earthquakes by distance
local = 0
regional = 0
distant = 0

for i in distance:
    if i < 50:
        local += 1
    elif 50 <= i < 150:
        regional +=1
    else:
        distant += 1

print("Local earthquakes:", local)
print("Regional earthquakes:", regional)
print("Distant earthquakes:", distant)

#plot of event vs distance
events_number = np.arange(1, len(distance) + 1)
plt.figure(figsize = (8, 5))
plt.plot(events_number, distance, marker = 'o')
plt.xlabel("Event Number")
plt.ylabel("Distance (km)")
plt.title("Earthquake Distance from Station")
plt.grid(True) #adding grid lines in the background
plt.savefig("event_vs_distance.png", dpi=300)
plt.show()

#histogram of distances
plt.figure(figsize=(8,5))

plt.hist(distance, bins=10)

plt.xlabel("Distance (km)")
plt.ylabel("Number of Events")
plt.title("Distribution of Earthquake Distances")
plt.grid(True)

plt.savefig("distance_histogram.png", dpi=300)
plt.show()

#p-s difference v/s distance
plt.figure(figsize=(8,5))

plt.scatter(ps_difference, distance)

plt.xlabel("P-S Time Difference (s)")
plt.ylabel("Distance (km)")
plt.title("Relationship Between P-S Difference and Distance")
plt.grid(True)

plt.savefig("ps_vs_distance.png", dpi=300)
plt.show()
