import numpy as np
import MLAProject.Knn.kNN as knn
import os

TRAIN_SET = "./res/digits/trainingDigits/"
TEST_SET = "./res/digits/testDigits/"


def img2vec(filename):
    return_vec = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            return_vec[0, i * 32 + j] = int(line[j])
    return return_vec


def my_way():
    class_label = []
    train_data = os.listdir(TRAIN_SET)
    train_data_count = len(train_data)
    train_data_mat = np.zeros((train_data_count, 1024))
    error_count = 0
    for index, i in enumerate(train_data):
        label = int(i.split('.')[0].split('_')[0])
        class_label.append(label)
        train_data_mat[index] = img2vec(os.path.join(TRAIN_SET, i))
    test_data = os.listdir(TEST_SET)
    test_data_count = len(test_data)
    for i in test_data:
        label = int(i.split('.')[0].split('_')[0])
        result = knn.classify(img2vec(os.path.join(TEST_SET, i)), np.array(train_data_mat), class_label, 3)
        if result != label:
            error_count += 1
            print(result, label)
            print("your classifier is %d , the real result is %d" % (result, label))
    print("the digist recognition correct rate is %{}".format((test_data_count - error_count) * 100 / test_data_count))


if __name__ == "__main__":
    my_way()