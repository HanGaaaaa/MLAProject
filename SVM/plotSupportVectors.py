from numpy import *
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

FILE_PATH = './res/testSet.txt'

x_cord0 = []
y_cord0 = []
x_cord1 = []
y_cord1 = []
markers = []
colors = []
fr = open(FILE_PATH)
for line in fr.readlines():
    lineSplit = line.strip().split('\t')
    xPt = float(lineSplit[0])
    yPt = float(lineSplit[1])
    label = int(lineSplit[2])
    if label == -1:
        x_cord0.append(xPt)
        y_cord0.append(yPt)
    else:
        x_cord1.append(xPt)
        y_cord1.append(yPt)

fr.close()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_cord0, y_cord0, marker='*', s=90)
ax.scatter(x_cord1, y_cord1, marker='o', s=50, c='red')
plt.title('Support Vectors Circled')
circle = Circle((4.6581910000000004, 3.507396), 0.5, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
ax.add_patch(circle)
circle = Circle((3.4570959999999999, -0.082215999999999997), 0.5, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
ax.add_patch(circle)
circle = Circle((6.0805730000000002, 0.41888599999999998), 0.5, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
ax.add_patch(circle)
b = -3.75567
w0 = 0.8065
w1 = -0.2761
x = arange(-2.0, 12.0, 0.1)
y = (-w0*x - b)/w1
ax.plot(x, y)
ax.axis([-2, 12, -8, 6])
plt.show()