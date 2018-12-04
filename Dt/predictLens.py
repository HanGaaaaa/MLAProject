import MLAProject.Dt.DecisionTree as dt
import MLAProject.Dt.treePlotter as tp

FILE_NAME = './res/lenses.txt'


def load_data(file_name):
    file = open(file_name)
    data_set = []
    feature_label = ['age', 'prescript', 'astigmatic', 'tearRate']
    for line in file.readlines():
        line = line.strip().split('\t')
        data_set.append(line)
    return data_set, feature_label


if __name__ == '__main__':
    data, labels = load_data(FILE_NAME)
    print(data)
    decision_tree = dt.create_tree(data, labels)
    print(decision_tree)
    result = dt.classify(decision_tree, labels, ['young', 'myope', 'no', 'reduced'])
    print(result)
    tp.create_plot(decision_tree)