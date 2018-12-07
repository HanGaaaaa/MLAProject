import MLAProject.Bayes.naiveBayes as nb
import re
import random

SPAM_PATH = './res/mail/spam/%d.txt'
HAM_PATH = './res/mail/ham/%d.txt'


def text_parse(big_string):
    list_of_tokens = re.split(r'\w*', big_string)
    return [token.lower() for token in list_of_tokens if len(token) > 2]


def mail_classify_test(cv=5):
    doc_list = []
    class_list = []
    for i in range(1, 26):
        print(i)
        doc_list.append(text_parse(open(SPAM_PATH % i).read()))
        class_list.append(1)
        doc_list.append(text_parse(open(HAM_PATH % i).read()))
        class_list.append(0)
    vocab_list = nb.create_vocab_database(doc_list)
    total_cv_err_rate = 0
    for times in range(cv):
        train_set = [i for i in range(50)]
        test_set = []
        for i in range(10):
            rand_index = int(random.uniform(0, len(train_set)))
            test_set.append(train_set[rand_index])
            del (train_set[rand_index])
        train_mat = []
        train_class = []
        for i in train_set:
            train_mat.append(nb.input_transform_vocab(vocab_list, doc_list[i]))
            train_class.append(class_list[i])
        pw, pc = nb.train_naive_bayes(train_mat, train_class)
        err_count = 0
        for i in range(len(test_set)):
            result = nb.classify(pw, pc, nb.input_transform_vocab(vocab_list, doc_list[i]))
            if result != class_list[i]:
                print('error mail:', test_set[i])
                err_count += 1
        err_rate = err_count / float(len(test_set))
        print('错误率：%.2f' % err_rate)
        total_cv_err_rate += err_rate
    total_cv_err_rate = total_cv_err_rate / cv
    print('交叉验证%d次后的错误率为：%.2f' % (cv, total_cv_err_rate))


if __name__ == '__main__':
    mail_classify_test()

