import numpy as np
import operator


def create_data_set():
    group = np.array([[1.0, 9999],
                      [1.0, 10003],
                      [0, 6000],
                      [0, 6300]])
    label = ['A', 'A', 'B', 'B']
    return group, label


def data_normal(data):
    min_val = data.min(0)
    max_val = data.max(0)
    range_val = max_val - min_val
    data_shape = data.shape
    normal_data = data - np.tile(min_val, (data_shape[0], 1))
    normal_data = normal_data / np.tile(range_val, (data_shape[0], 1))
    return normal_data, min_val, range_val


def classify(test_data, data_set, label, k):
    data_set_size = data_set.shape[0]
    diff_mat = np.tile(test_data, (data_set_size, 1)) - data_set
    instances = np.sum(diff_mat ** 2, axis=1) ** 0.5
    sort_mat = instances.argsort()
    class_count = {}
    for i in range(k):
        class_label = label[sort_mat[i]]
        class_count[class_label] = class_count.get(class_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


if __name__ == '__main__':
    data, label = create_data_set()
    new_data, min_v, range_v = data_normal(data)
    test = np.array([[0.4, 10000]])
    new_test = (test - min_v) / range_v
    result = classify(new_test, new_data, label, 3)
    print(result)