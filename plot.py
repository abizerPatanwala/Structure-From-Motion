#!/usr/bin/env python3

import matplotlib.pyplot as plt

def main():
  xPts = []
  zPts = []
  with open('world_pts.txt') as f:
    for line in f:
      xPts.append(float(line.strip().split(',')[0])) 
      zPts.append(float(line.strip().split(',')[2])) 

  fig, ax = plt.subplots()
  ax.scatter(xPts, zPts, s = 1)
  plt.show()
  ax.set(xlim=(-15, 15),ylim=(-5, 25))






if __name__ == "__main__":
    main()
