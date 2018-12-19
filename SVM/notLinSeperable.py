from numpy import *
import matplotlib.pyplot as plt

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
    if label == 0:
        x_cord0.append(xPt)
        y_cord0.append(yPt)
    else:
        x_cord1.append(xPt)
        y_cord1.append(yPt)

fr.close()
fig = plt.figure()
ax = fig.add_subplot(221)
x_cord0 = []
y_cord0 = []
x_cord1 = []
y_cord1 = []
for i in range(300):
    [x, y] = random.uniform(0, 1, 2)
    if ((x > 0.5) and (y < 0.5)) or ((x < 0.5) and (y > 0.5)):
        x_cord0.append(x)
        y_cord0.append(y)
    else:
        x_cord1.append(x)
        y_cord1.append(y)
ax.scatter(x_cord0, y_cord0, marker='.', s=90)
ax.scatter(x_cord1, y_cord1, marker='*', s=50, c='red')
plt.title('A')
ax = fig.add_subplot(222)
x_cord0 = random.standard_normal(150)
y_cord0 = random.standard_normal(150)
x_cord1 = random.standard_normal(150) + 2.0
y_cord1 = random.standard_normal(150) + 2.0
ax.scatter(x_cord0, y_cord0, marker='.', s=90)
ax.scatter(x_cord1, y_cord1, marker='*', s=50, c='red')
plt.title('B')
ax = fig.add_subplot(223)
x_cord0 = []
y_cord0 = []
x_cord1 = []
y_cord1 = []
for i in range(300):
    [x, y] = random.uniform(0, 1, 2)
    if x > 0.5:
        x_cord0.append(x * cos(2.0 * pi * y))
        y_cord0.append(x * sin(2.0 * pi * y))
    else:
        x_cord1.append(x * cos(2.0 * pi * y))
        y_cord1.append(x * sin(2.0 * pi * y))
ax.scatter(x_cord0, y_cord0, marker='.', s=90)
ax.scatter(x_cord1, y_cord1, marker='*', s=50, c='red')
plt.title('C')
ax = fig.add_subplot(224)

x_cord1 = zeros(150)
y_cord1 = zeros(150)
x_cord0 = random.uniform(-3, 3, 350)
y_cord0 = random.uniform(-3, 3, 350)
x_cord1[0:50] = 0.3 * random.standard_normal(50) + 2.0
y_cord1[0:50] = 0.3 * random.standard_normal(50) + 2.0

x_cord1[50:100] = 0.3 * random.standard_normal(50) - 2.0
y_cord1[50:100] = 0.3 * random.standard_normal(50) - 3.0

x_cord1[100:150] = 0.3 * random.standard_normal(50) + 1.0
y_cord1[100:150] = 0.3 * random.standard_normal(50)

ax.scatter(x_cord0, y_cord0, marker='.', s=90)
ax.scatter(x_cord1, y_cord1, marker='*', s=50, c='red')
plt.title('D')
plt.show()