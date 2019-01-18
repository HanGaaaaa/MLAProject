import numpy as np
import MLAProject.AdaBoost.adaboost as ab


file_path_train = './res/horseColicTraining2.txt'
file_path_test = './res/horseColicTest2.txt'


def load_data(file_name):
    data_mat = []
    label_mat = []
    fr = open(file_name)
    num_feat = len(fr.readline().split('\t'))
    for i in fr.readlines():
        line_arr = []
        cur_line = i.strip().split('\t')
        for j in range(num_feat - 1):
            line_arr.append(float(cur_line[j]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


if __name__ == '__main__':
    data_arr, label_arr = load_data(file_path_train)
    classifier = ab.boost_train_ds(data_arr, label_arr, 50)
    print(classifier)
    test_arr, teat_label = load_data(file_path_test)
    test_count = len(test_arr)
    result = ab.ada_classify(test_arr, classifier)
    err_mat = np.mat(np.ones((test_count, 1)))
    err_count = err_mat[result != np.mat(teat_label).T].sum()
    err_rate = err_count / test_count
    print(err_rate)

