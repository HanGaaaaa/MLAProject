import operator
from math import log


def create_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def calc_shannon_ent(data_set):
    data_count = len(data_set)
    label_count = {}
    for sample in data_set:
        label = sample[-1]
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
        shannon_ent = 0.0
    for label_class in label_count:
        label_p = label_count[label_class] / data_count
        single_ent = label_p * log(label_p, 2)
        shannon_ent -= single_ent
    return shannon_ent


def split_data_set(data, axis, value):
    ret_data_set = []
    for sample in data:
        if sample[axis] == value:
            reduce_data = sample[0:axis]
            reduce_data.extend(sample[axis + 1:])
            ret_data_set.append(reduce_data)
    return ret_data_set


def chose_best_feature_to_split(data):
    feature_count = len(data[0]) - 1
    base_ent = calc_shannon_ent(data)
    best_ent = 0.0
    best_feature = -1
    for i in range(feature_count):
        feature_value = [sample[i] for sample in data]
        feature_value = set(feature_value)
        new_ent = 0.0
        for value in feature_value:
            new_data = split_data_set(data, i, value)
            new_data_p = len(new_data) / len(data)
            value_ent = new_data_p * calc_shannon_ent(new_data)
            new_ent += value_ent
        gain_ent = base_ent - new_ent
        if gain_ent > best_ent:
            best_ent = gain_ent
            best_feature = i
    return best_feature


def majority_class(class_list):
    class_count = {}
    for i in class_list:
        if i not in class_count:
            class_count[i] = 0
        class_count[i] += 1
    sorted_class_count = sorted(class_count, key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data, label):
    fun_label = label[:]
    class_label = [sample[-1] for sample in data]
    if class_label.count(class_label[0]) == len(class_label):
        return class_label[0]
    if len(data[0]) == 1:
        return majority_class(class_label)
    best_feature = chose_best_feature_to_split(data)
    best_label = fun_label[best_feature]
    my_tree = {best_label: {}}
    del(fun_label[best_feature])
    feature_value = [sample[best_feature] for sample in data]
    feature_value = set(feature_value)
    for value in feature_value:
        #sub_label = label[:]
        my_tree[best_label][value] = create_tree(split_data_set(data, best_feature, value), fun_label)
    return my_tree


def classify(tree_dic, feat_label, input_vec):
    first_key = list(tree_dic.keys())[0]
    feature_index = feat_label.index(first_key)
    second_dic = tree_dic[first_key]
    for key in second_dic.keys():
        if input_vec[feature_index] == key:
            if type(second_dic[key]).__name__ == 'dict':
                return classify(second_dic[key], feat_label, input_vec)
            else:
                return second_dic[key]


if __name__ == "__main__":
    data_set, labels = create_data_set()
    decision_tree = create_tree(data_set, labels)
    print(decision_tree)
    print(labels)
    result = classify(decision_tree, labels, [1, 1])
    print(result)
