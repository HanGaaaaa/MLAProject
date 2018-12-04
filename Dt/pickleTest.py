import pickle
import MLAProject.Dt.DecisionTree as dt
import time

FILE_NAME = './res/model.txt'


def save_tree(tree, file_name):
    fw = open(file_name, 'wb+')
    pickle.dump(tree, fw)
    fw.close()


def load_tree(file_name):
    fr = open(file_name, 'rb+')
    return pickle.load(fr)


if __name__ == '__main__':
    data_set, labels = dt.create_data_set()
    decision_tree = dt.create_tree(data_set, labels)
    print(decision_tree)
    save_tree(decision_tree, FILE_NAME)
    time.sleep(3)
    model = load_tree(FILE_NAME)
    result = dt.classify(model, labels, [1, 1])
    print(result)