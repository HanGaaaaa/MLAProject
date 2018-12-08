import numpy as np
import matplotlib.pyplot as plt
import random
import warnings

warnings.filterwarnings('ignore')
FILE_PATH = './res/testSet.txt'


def load_data_set(file_name):
    x_data = []
    y_data = []
    for line in open(file_name).readlines():
        line.strip()
        lina_arr = line.split()
        x_data.append([1.0, float(lina_arr[0]), float(lina_arr[1])])
        y_data.append(int(lina_arr[2]))
    return x_data, y_data


def sigmoid(input_x):
    return 1.0 / (1 + np.exp(-input_x))


def draw(x_data, y_data, get_weight):
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    n = np.shape(x_data)[0]
    plt.figure(facecolor='w')
    for i in range(n):
        if int(y_data[i]) == 1:
            plt.scatter(x_data[i, 1], x_data[i, 2], marker='.', c='g')
        else:
            plt.scatter(x_data[i, 1], x_data[i, 2], marker='.', c='r')
    x1 = np.linspace(-3, 3, 600)
    print(x1)
    x2 = (-get_weight[0] - get_weight[1] * x1) / get_weight[2]
    plt.plot(x1, x2)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def gradient_ascent_train(train_data, train_label, circle=500, alpha=0.001):
    train_data_mat = np.mat(train_data)
    train_label = np.mat(train_label).transpose()
    m, n = train_data_mat.shape
    train_weight = np.ones((n, 1))
    for i in range(circle):
        y_hat = sigmoid(train_data_mat * train_weight)
        error = train_label - y_hat
        train_weight += alpha * train_data_mat.transpose() * error
    return train_weight


def random_gradient_ascent_train(train_data, train_label, circle=150):
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    m, n = train_data.shape
    train_weight = np.ones(n)
    w0_change_line = []
    w1_change_line = []
    w2_change_line = []
    cur_circle = 0
    for i in range(circle):
        print('当前第 %d次第迭代' % i)
        data_index = [index for index in train_data]
        for j in range(m):
            cur_circle += 1
            alpha = 4 / (1.0 + i + j) + 0.01
            random_index = int(random.uniform(0, len(data_index)))
            y_hat = sigmoid(np.sum(train_data[random_index] * train_weight))
            error = train_label[random_index] - y_hat
            train_weight += alpha * error * train_data[random_index]
            if cur_circle % 100 == 0:
                w0_change_line.append([cur_circle, train_weight[0]])
                w1_change_line.append([cur_circle, train_weight[1]])
                w2_change_line.append([cur_circle, train_weight[2]])
            del(data_index[random_index])
    return train_weight, w0_change_line, w1_change_line, w2_change_line


def draw_w_change_convergence(w0_change_line, w1_change_line, w2_change_line):
    plt.figure(facecolor='w')
    w0_change_line = np.array(w0_change_line)
    w1_change_line = np.array(w1_change_line)
    w2_change_line = np.array(w2_change_line)
    plt.subplot(3, 1, 1)
    plt.plot(w0_change_line[:, 0], w0_change_line[:, 1], '-', lw=2)
    plt.subplot(3, 1, 2)
    plt.plot(w1_change_line[:, 0], w1_change_line[:, 1], '-', lw=2)
    plt.subplot(3, 1, 3)
    plt.plot(w2_change_line[:, 0], w2_change_line[:, 1], '-', lw=2)
    plt.show()


def classify(input_data, w):
    value = sigmoid(np.sum(input_data * w))
    if value > 0.5:
        return 1
    else:
        return 0


if __name__ == '__main__':
    data, label = load_data_set(FILE_PATH)
    # weight = gradient_ascent_train(data, label)
    weight, w_change1, w_change2, w_change3 = random_gradient_ascent_train(data, label, circle=100)
    draw_w_change_convergence(w_change1, w_change2, w_change3)
    draw(data, label, weight)