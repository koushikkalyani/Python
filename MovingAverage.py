import numpy as np
from itertools import islice
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

#Function to find moving avaerage ... 
def MovingAverage(a,period):
  iteration = len(a) - period + 1
  result = np.zeros(iteration,dtype=np.uint8)
  for i in range(iteration):
      sum = 0
      for j in range(period):
       sum += a[i+j]
      result[i] = sum // period
  return np.array(result)

data=np.random.randint(10, size=(50))
period = 2
M_average=MovingAverage(data,period)

data_points = []
moving_avg = []

# Create a figure and axis
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

# Simulate a for loop
for i in range(len(M_average)):
    data_points.append(data[i])       # Add current iteration to x-values
    moving_avg.append(M_average[i])    # Example: y = i^2
    
    ax.clear()  # Clear the previous plot
    ax.plot(data_points,  marker='o', color='b')  # Plot updated data
    ax.plot(moving_avg, marker= '*', color = 'r')
    plt.xlabel("Time Series")
    plt.ylabel("Data Points")
    plt.title("Moving Average")
    
    plt.draw()  # Redraw the figure
    plt.pause(0.01)  # Pause for a short duration (e.g., 0.5 seconds)
    
    time.sleep(0.01)  # Simulate some computation delay

plt.ioff()  # Turn off interactive mode
plt.show()


