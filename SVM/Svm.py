import numpy as np

FILE_PATH = './res/testSet.txt'
RBF_FILE_PATH = './res/testSetRBF.txt'
RBF_TEST_FILE_PATH = './res/testSetRBF2.txt'


def load_data(file_path):
    data_mat = []
    label_mat = []
    file = open(file_path)
    for i in file.readlines():
        file_arr = i.strip().split('\t')
        data_mat.append([float(file_arr[0]), float(file_arr[1])])
        label_mat.append(float(file_arr[2]))
    return data_mat, label_mat


def select_j_rand(i, m):
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j


def clip_alpha(aj, high, low):
    if aj > high:
        aj = high
    if aj < low:
        aj = low
    return aj


def sample_smo(data_mat_in, data_label_in, c, toler, max_iter):
    data_matrix = np.mat(data_mat_in)
    data_label = np.mat(data_label_in).transpose()
    b = 0
    m, n = data_matrix.shape
    alpha = np.mat(np.zeros((m, 1)))
    iter_count = 0
    while iter_count < max_iter:
        alpha_pairs_changed = 0
        for i in range(m):
            fxi = float(np.multiply(alpha, data_label).T * (data_matrix * data_matrix[i].T)) + b
            Ei = fxi - float(data_label[i])
            if (data_label[i] * Ei > toler and alpha[i] > 0) or \
                    (data_label[i] * Ei < -toler and alpha[i] < c):
                j = select_j_rand(i, m)
                fxj = float(np.multiply(alpha, data_label).T * (data_matrix * data_matrix[j].T)) + b
                Ej = fxj - float(data_label[j])
                alpha_i_old = alpha[i].copy()
                alpha_j_old = alpha[j].copy()
                if data_label[i] == data_label[j]:
                    low = max(0, alpha[i] + alpha[j] - c)
                    high = min(alpha[i] + alpha[j], c)
                elif data_label[i] != data_label[j]:
                    low = max(0, alpha[j] - alpha[i])
                    high = min(c, c + alpha[j] - alpha[i])
                if low == high:
                    continue
                eta = 2.0 * data_matrix[i] * data_matrix[j].T \
                    - data_matrix[i] * data_matrix[i].T \
                    - data_matrix[j] * data_matrix[j].T
                if eta > 0:
                    continue
                alpha[j] -= data_label[j] * (Ei - Ej) / eta
                alpha[j] = clip_alpha(alpha[j], high, low)
                if abs(alpha[j] - alpha_j_old) < 0.00001:
                    continue
                alpha[i] += data_label[i] * data_label[j] * (alpha_j_old - alpha[j])
                b1 = b - Ei \
                     - data_label[i] * data_matrix[i] * data_matrix[i].T * (alpha[i] - alpha_i_old) \
                     - data_label[j] * data_matrix[i] * data_matrix[j].T * (alpha[j] - alpha_j_old)
                b2 = b - Ej \
                     - data_label[i] * data_matrix[i] * data_matrix[j].T * (alpha[i] - alpha_i_old) \
                     - data_label[j] * data_matrix[j] * data_matrix[j].T * (alpha[j] - alpha_j_old)
                if 0 < alpha[i] < c:
                    b = b1
                elif 0 < alpha[j] < c:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
        if alpha_pairs_changed == 0:
            iter_count += 1
        else:
            iter_count = 0
    return alpha, b


class OptionStruct:
    def __init__(self, data_mat_in, class_labels, c, toler, tup):
        self.x = data_mat_in
        self.label = class_labels
        self.c = c
        self.toler = toler
        self.m = data_mat_in.shape[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.e_cache = np.mat(np.zeros((self.m, 2)))
        self.k = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.k[:, i] = kernel_trans(self.x, self.x[i], tup)


def calc_ek(os, k):
    fxk = float(np.multiply(os.alphas, os.label).T * os.k[:, k] + os.b)
    ek = fxk - float(os.label[k])
    return ek


def select_j(i, os, ei):
    max_k = -1
    max_delta_e = 0
    ej = 0
    os.e_cache[i] = [1, ei]
    valid_e_cache = np.nonzero(os.e_cache[:, 0].A)[0]
    if (len(valid_e_cache)) > 1:
        for k in valid_e_cache:
            if k == i:
                continue
            ek = calc_ek(os, k)
            delta_e = np.abs(ei - ek)
            if delta_e > max_delta_e:
                max_k = k
                max_delta_e = delta_e
                ej = ek
        return max_k, ej
    else:
        j = select_j_rand(i, os.m)
        ej = calc_ek(os, j)
    return j, ej


def update_ek(os, k):
    ek = calc_ek(os, k)
    os.e_cache[k] = [1, ek]


def inner_loop(i, os):
    ei = calc_ek(os, i)
    if ((os.label[i] * ei < -os.toler) and (os.alphas[i] < os.c)) or (
            (os.label[i] * ei > os.toler) and (os.alphas[i] > 0)):
        j, ej = select_j(i, os, ei)
        alpha_i_old = os.alphas[i].copy()
        alpha_j_old = os.alphas[j].copy()
        if os.label[i] != os.label[j]:
            low = max(0, os.alphas[j] - os.alphas[i])
            high = min(os.c, os.c + os.alphas[j] - os.alphas[i])
        else:
            low = max(0, os.alphas[j] + os.alphas[i] - os.c)
            high = min(os.c, os.alphas[j] + os.alphas[i])
        # if low == high:
        #     print('low == high')
        #     return 0
        eta = 2.0 * os.k[i, j] - os.k[i, i] - os.k[j, j]
        # if eta >= 0:
        #     print('eat >= 0')
        #     return 0
        os.alphas[j] -= os.label[j] * (ei - ej) / eta
        os.alphas[j] = clip_alpha(os.alphas[j], high, low)
        update_ek(os, j)
        if abs(os.alphas[j] - alpha_j_old) < 0.00001:
            print('j not moving enough')
            return 0
        os.alphas[i] += os.label[j] * os.label[i] * (alpha_j_old - os.alphas[j])
        update_ek(os, i)
        b1 = os.b - ei - \
            os.label[i] * (os.alphas[i] - alpha_i_old) * os.k[i, i] - \
            os.label[j] * (os.alphas[j] - alpha_j_old) * os.k[i, j]
        b2 = os.b - ej - \
            os.label[i] * (os.alphas[i] - alpha_i_old) * os.k[i, j] - \
            os.label[j] * (os.alphas[j] - alpha_j_old) * os.k[j, j]
        if (0 < os.alphas[i]) and (os.c > os.alphas[i]):
            os.b = b1
        elif (0 < os.alphas[j]) and (os.c > os.alphas[j]):
            os.b = b2
        else:
            os.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smo_complete(data_mat_in, class_labels, c, toler, max_iter, k_tup=('lin', 0)):
    os = OptionStruct(np.mat(data_mat_in), np.mat(class_labels).transpose(), c, toler, k_tup)
    iter_count = 0
    entire_set = True
    alpha_pairs_changed = 0
    while (iter_count < max_iter) and (alpha_pairs_changed > 0 or entire_set):
        alpha_pairs_changed = 0
        if entire_set:
            for i in range(os.m):
                alpha_pairs_changed += inner_loop(i, os)
                print('full set iter: %d i:%d, pairs changed %d' % (iter_count, i, alpha_pairs_changed))
            iter_count += 1
        else:
            non_bound_i = np.nonzero((os.alphas.A > 0) * (os.alphas.A < os.c))[0]
            for i in non_bound_i:
                alpha_pairs_changed += inner_loop(i, os)
                print('nonbounds iter: %d i: %d, pairs changed %d' % (iter_count, i, alpha_pairs_changed))
            iter_count += 1
        if entire_set:
            entire_set = False
        elif alpha_pairs_changed == 0:
            entire_set = True
        print('iteration number: %d' % iter_count)
    return os.b, os.alphas


def calc_w(alphas, data_array, class_label):
    x = np.mat(data_array)
    label = np.mat(class_label).transpose()
    m, n = x.shape
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * label[i], x[i].T)
    return w


def kernel_trans(x, a, tup):
    m, n = np.shape(x)
    k = np.mat(np.zeros((m, 1)))
    if tup[0] == 'lin':
        k = x * a.T
    elif tup[0] == 'rbf':
        for i in range(m):
            delta_row = x[i] - a
            k[i] = delta_row * delta_row.T
        k = np.exp(k / (-1 * tup[1] ** 2))
    else:
        raise NameError('kernel not command')
    return k


if __name__ == '__main__':
    # data, label = load_data(RBF_FILE_PATH)
    # b, alphas = smo_complete(data, label, 200, 0.001, 40, ('rbf', 1.3))
    # ws = calc_w(alphas, data, label)
    # print(ws)
    # data_mat = np.mat(data)
    # result = data_mat[0] * np.mat(ws) + b
    # print(np.sign(result))
    # print(label[0])
    data_arr, label_arr = load_data(RBF_FILE_PATH)
    b, alphas = smo_complete(data_arr, label_arr, 200, 0.0001, 10000, ('rbf', 1.3))
    dat_mat = np.mat(data_arr)
    label_mat = np.mat(label_arr).transpose()
    valid_index = np.nonzero(alphas.A > 0)[0]
    valid_data = dat_mat[valid_index]
    valid_label = label_mat[valid_index]
    print('there are %d support vectors' % np.shape(valid_data)[0])
    m, n = np.shape(dat_mat)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(valid_data, dat_mat[i], ('rbf', 1.3))
        predict = kernel_eval.T * np.multiply(valid_label, alphas[valid_index]) + b
        if np.sign(predict) != np.sign(label_arr[i]):
            error_count += 1
    print('the training error rate is %d / %d = %f ' % (error_count, m, (float(error_count) / m)))
    dataArr, labelArr = load_data(RBF_TEST_FILE_PATH)
    errorCount = 0
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernel_trans(valid_data, datMat[i, :], ('rbf', 1.3))
        predict = kernelEval.T * np.multiply(valid_label, alphas[valid_index]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))

