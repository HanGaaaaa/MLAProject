import numpy as np


def load_simple_data():
    data_mat = np.matrix([[1., 2.1],
                          [2., 1.1],
                          [1.3, 1.],
                          [1., 1.],
                          [2., 1.]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, class_labels


def stump_classify(data_matrix, dimen, thresh_val, thresh_ineq):
    ret_array = np.ones((np.shape(data_matrix)[0], 1))
    if thresh_ineq == 'lt':
        ret_array[data_matrix[:, dimen] <= thresh_val] = -1.0
    else:
        ret_array[data_matrix[:, dimen] > thresh_val] = -1.0
    return ret_array


def build_stump(data_mat, class_labels, D):
    data = np.mat(data_mat)
    labels = np.mat(class_labels).T
    m, n = np.shape(data)
    step_count = 10
    best_stump = {}
    best_class_est = np.mat(np.zeros((m, 1)))
    min_err = np.inf
    for i in range(n):
        range_min = data[:, i].min()
        range_max = data[:, i].max()
        step_size = (range_max - range_min) / step_count
        for j in range(-1, int(step_count) + 1):
            for inequel in ['lt', 'gt']:
                thresh_val = range_min + float(j) * step_size
                predicted_vals = stump_classify(data, i, thresh_val, inequel)
                err_arr = np.mat(np.ones((m, 1)))
                err_arr[predicted_vals == labels] = 0
                weight_err = D.T * err_arr
                print("split: dim %d, thresh: %.2f, thresh inequl: %s, the weight error is: %.3f" % (i, thresh_val, inequel, weight_err))
                if weight_err < min_err:
                    min_err = weight_err
                    best_class_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequel
    return best_stump, min_err, best_class_est


def boost_train_ds(data_arr, class_labels, num_iter=40):
    week_class_arr = []
    m = np.shape(data_arr)[0]
    D = np.mat(np.ones((m, 1)) / m)
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(num_iter):
        best_stump, min_err, class_est = build_stump(data_arr, class_labels, D)
        print('D: ', D.T)
        print('error: ', min_err)
        alpha = float(0.5 * np.log((1.0 - min_err) / max(min_err, 1e-16)))
        print('alpha: ', alpha)
        best_stump['alpha'] = alpha
        week_class_arr.append(best_stump)
        print('class est:', class_est)
        expon = np.multiply(-1 * alpha * np.mat(class_labels).T, class_est)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        agg_class_est += class_est * alpha
        print('aggclassest: ', agg_class_est)
        agg_error = np.multiply(np.sign(agg_class_est) != np.mat(class_labels).T, np.ones((m, 1)))
        error_rate = agg_error.sum() / m
        if error_rate == 0.0:
            break
    return week_class_arr


def ada_classify(data_to_class, classifyer):
    data = np.mat(data_to_class)
    m = np.shape(data)[0]
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(len(classifyer)):
        class_est = stump_classify(data, classifyer[i]['dim'], classifyer[i]['thresh'], classifyer[i]['ineq'])
        agg_class_est += class_est * classifyer[i]['alpha']
        print(agg_class_est)
        return np.sign(agg_class_est)


if __name__ == '__main__':
    data_train, label_train = load_simple_data()
    classify_array = boost_train_ds(data_train, label_train)
    print(classify_array)
    result = ada_classify([0, 0], classify_array)
    print(result)