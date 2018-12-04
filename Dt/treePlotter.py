import matplotlib.pyplot as plt
from pylab import mpl
import MLAProject.Dt.DecisionTree as dt

mpl.rcParams['font.sans-serif'] = ['SimHei']
decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plot_node(node_txt, center_pt, parent_pt, node_type):
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction',
                             xytext=center_pt, textcoords='axes fraction',
                             va="center", ha="center", bbox=node_type, arrowprops=arrow_args)


def plot_mid_text(cnt_pt, parent_pt, txt):
    x_mid = (parent_pt[0] - cnt_pt[0]) / 2.0 + cnt_pt[0]
    y_mid = (parent_pt[1] - cnt_pt[1]) / 2.0 + cnt_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt)


def create_plot(in_tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.total_width = float(get_leaf_count(in_tree))
    plot_tree.total_depth = float(get_tree_depth(in_tree))
    plot_tree.x_off = -0.5 / plot_tree.total_width
    plot_tree.y_off = 1.0
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()


def plot_tree(my_tree, parent_pt, node_text):
    leaf_count = get_leaf_count(my_tree)
    first_key = list(my_tree.keys())[0]
    cnt_pt = (plot_tree.x_off + (1.0 + float(leaf_count)) / 2.0 / plot_tree.total_width, plot_tree.y_off)
    plot_mid_text(cnt_pt, parent_pt, node_text)
    plot_node(first_key, cnt_pt, parent_pt, decision_node)
    second_dic = my_tree[first_key]
    plot_tree.y_off = plot_tree.y_off - 1.0 / plot_tree.total_depth
    for key in second_dic.keys():
        if type(second_dic[key]).__name__ == 'dict':
            plot_tree(second_dic[key], cnt_pt, str(key))
        else:
            plot_tree.x_off = plot_tree.x_off + 1.0 / plot_tree.total_width
            plot_node(second_dic[key], (plot_tree.x_off, plot_tree.y_off), cnt_pt, leaf_node)
            plot_mid_text((plot_tree.x_off, plot_tree.y_off), cnt_pt, str(key))
    plot_tree.y_off = plot_tree.y_off + 1.0 / plot_tree.total_depth


def my_get_leaf_count(tree_dic):
    leaf_count = 0
    print(tree_dic)
    for i in tree_dic:
        if type(tree_dic[i]).__name__ == 'dict':
            leaf_count += my_get_leaf_count(tree_dic[i])
        else:
            leaf_count += 1
    return leaf_count


def get_leaf_count(tree_dic):
    leaf_count = 0
    first_key = list(tree_dic.keys())[0]
    second_dic = tree_dic[first_key]
    for i in second_dic.keys():
        if type(second_dic[i]).__name__ == 'dict':
            leaf_count += get_leaf_count(second_dic[i])
        else:
            leaf_count += 1
    return leaf_count


def get_tree_depth(tree_dic):
    tree_depth = 0
    first_key = list(tree_dic.keys())[0]
    second_dic = tree_dic[first_key]
    for i in second_dic.keys():
        if type(second_dic[i]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dic[i])
        else:
            this_depth = 1
        if this_depth > tree_depth:
            tree_depth = this_depth
    return tree_depth


if __name__ == "__main__":
    data_set, labels = dt.create_data_set()
    decision_tree = dt.create_tree(data_set, labels)
    print(decision_tree)
    create_plot(decision_tree)



