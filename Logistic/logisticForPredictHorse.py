import MLAProject.Logistic.logistRegression as lg

TRAIN_FILE_PATH = './res/horseColicTraining.txt'
TEST_FILE_PATH = './res/horseColicTest.txt'


def load_data():
    train_data = []
    train_label = []
    test_data = []
    teat_label = []
    for line in open(TRAIN_FILE_PATH).readlines():
        data_arr = line.strip().split()
        line_arr = []
        for i in range(21):
            line_arr.append(float(data_arr[i]))
        train_data.append(line_arr)
        train_label.append(float(data_arr[-1]))
    for line in open(TEST_FILE_PATH).readlines():
        data_arr = line.strip().split()
        line_arr = []
        for i in range(21):
            line_arr.append(float(data_arr[i]))
        test_data.append(line_arr)
        teat_label.append(float(data_arr[-1]))
    return train_data, train_label, test_data, teat_label


if __name__ == '__main__':
    train_data, train_label, test_data, teat_label = load_data()
    w, w0, w1, w2 = lg.random_gradient_ascent_train(train_data, train_label)
    # w = lg.gradient_ascent_train(train_data, train_label)
    err_count = 0
    m = len(test_data)
    for i in range(m):
        if teat_label[i] != lg.classify(test_data[i], w):
            err_count += 1
    correct = (m - err_count) / m * 100
    print('测试的正确率为 %.2f%% ' % correct)
    # lg.draw_w_change_convergence(w0, w1, w2)
