from numpy import *
import matplotlib
import matplotlib.pyplot as plt

RBF_FILE_PATH = './res/testSetRBF.txt'
fw = open(RBF_FILE_PATH, 'w')

fig = plt.figure()
ax = fig.add_subplot(111)
x_cord0 = []
y_cord0 = []
x_cord1 = []
y_cord1 = []
for i in range(100):
    [x, y] = random.uniform(0, 1, 2)
    xpt = x * cos(2.0*pi*y)
    ypt = x*sin(2.0*pi*y)
    if x > 0.5:
        x_cord0.append(xpt)
        y_cord0.append(ypt)
        label = -1.0
    else:
        x_cord1.append(xpt)
        y_cord1.append(ypt)
        label = 1.0
    fw.write('%f\t%f\t%f\n' % (xpt, ypt, label))
ax.scatter(x_cord0, y_cord0, marker='s', s=90)
ax.scatter(x_cord1, y_cord1, marker='o', s=50, c='red')
plt.title('Non-linearly Separable Data for Kernel Method')
plt.show()
fw.close()