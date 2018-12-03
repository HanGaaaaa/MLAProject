import numpy as np
import MLAProject.Knn.kNN as knn

FILE_PATH = "./res/datingTestSet.txt"
TEST_FILE_PATH = "./res/datingTestSet2.txt"


def file2matrix(filename):
    fr = open(filename)
    lines = fr.readlines()
    line_len = len(lines)
    data_mat = np.zeros((line_len, 3))
    class_label = []
    for index, i in enumerate(lines):
        line = i.strip()
        list_from_line = line.split('\t')
        data_mat[index, :] = list_from_line[0:3]
        class_label.append(int(list_from_line[-1]))
    return data_mat, class_label


def dating_class():
    data, labels = file2matrix(TEST_FILE_PATH)
    norm_data, _, _ = knn.data_normal(data)
    data_spl = 0.1
    error_count = 0
    test_data_len = int(data.shape[0] * data_spl)
    for i in range(test_data_len):
        result = knn.classify(norm_data[i], norm_data[test_data_len:], labels[test_data_len:], 10)
        if result != labels[i]:
            print("thr classifer is %d, the real answer is %d" % (result, labels[i]))
            error_count += 1
    print(error_count)
    print("the total err rate is %f" % float((error_count / test_data_len)))


def classify_person():
    result_list = ['not at all', 'little', 'very like']
    percent_tats = float(input("percentage of time spent playing video games?"))
    f_miles = float(input("freguent flier miles earned per year?"))
    ice_cream = float(input("liters of ice cream consumed per year?"))
    test = [percent_tats, f_miles, ice_cream]
    data, labels = file2matrix(TEST_FILE_PATH)
    normal_data, min_v, range_v = knn.data_normal(data)
    result = knn.classify((test - min_v) / range_v, normal_data, labels, 10)
    print("you will %s" % result_list[result - 1])


if __name__ == '__main__':
    classify_person()
