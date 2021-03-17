# Implementation of matplotlib function 
import matplotlib.pyplot as plt 
import numpy as np 
import math
from scipy.integrate import simps

t = np.arange(0, 10, 0.01) 
y = t
plt.plot(t, y) 
plt.title('matplotlib.pyplot.ginput() function Example', fontweight ="bold") 

print("After 2 clicks :") 
x = plt.ginput(2) 
print(x) 
   
plt.clf()

endpoint1 = round(float(str(x[0]).split(', ')[0][1:]),2)
endpoint2 = round(float(str(x[1]).split(', ')[0][1:]),2)

print(endpoint1, endpoint2)

t2 = np.arange(endpoint1, endpoint2, 0.01)

#print(t, t2)

plt.close()

y2 = t2
plt.plot(t2, y2) 

area = simps(y2, dx = 0.01)
print(area)
print((pow(endpoint2, 2)-pow(endpoint1, 2))/2)

plt.show() 
