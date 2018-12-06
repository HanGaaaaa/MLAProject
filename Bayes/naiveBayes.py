import numpy as np


def load_data():
    data_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]
    return data_list, class_vec


def create_vocab_database(data_set):
    vocab_set = set()
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def input_transform_vocab(vocab_database, input_txt):
    vocab_vec = [0] * len(vocab_database)
    for word in input_txt:
        if word in vocab_database:
            vocab_vec[vocab_database.index(word)] += 1
    return vocab_vec


def article_2_word_bag(vocab_database, input_article):
    mat = []
    for row in input_article:
        mat.append(input_transform_vocab(vocab_database, row))
    return mat


def train_naive_bayes(data_mat, class_label):
    pcw = []
    denom = []
    pc = []
    data_mat_count = len(data_mat)
    word_num = len(data_mat[0])
    class_count = len(set(class_label))
    for i in range(class_count):
        pc.append(len([a for a in class_label if a == i]))
        pcw.append(np.ones(word_num))
        denom.append(2)
    pc = np.asarray(pc) / data_mat_count
    for i in range(data_mat_count):
        for j in range(class_count):
            if class_label[i] == j:
                pcw[j] += data_mat[i]
                denom[j] += sum(data_mat[i])
    for i in range(class_count):
        pcw[i] = np.log(pcw[i] / denom[i])
    return pcw, pc


def classify(pcw, pc, input_txt):
    p = []
    class_count = len(pc)
    for i in range(class_count):
        p.append(sum(input_txt * pcw[i]) + np.log(pc[i]))
    print(p)
    return p.index(sorted(p, reverse=True)[0])


if __name__ == '__main__':
    train_data, train_label = load_data()
    vocab_data = create_vocab_database(train_data)
    print(vocab_data)
    cw, c = train_naive_bayes(article_2_word_bag(vocab_data, train_data), train_label)
    print(cw[0])
    print('---------------------------------')
    print(cw[1])
    testEntry1 = ['love', 'my', 'dalmation']
    thisEntry2 = ['stupid', 'garbage']
    this = np.array(input_transform_vocab(vocab_data, thisEntry2))
    result = classify(cw, c, this)
    print(result)
