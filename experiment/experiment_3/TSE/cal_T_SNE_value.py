import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# 加载数据
def cal_T_SNE(dataset_name, model_name, data1, label1, data2, label2, save_path):
    type_list = ['Original', 'Retrained']
    # type_list = ['Retrained', 'Original']
    ts = TSNE(n_components=2, init='pca', random_state=0)

    for num in range(len(type_list)):
        tittle = type_list[num] + '_' + dataset_name + '_' + model_name
        print(type_list[num])
        if num == 0:
            data = data1
            label = label1
        else:
            data = data2
            label = label2
        result = ts.fit_transform(data)
        # 调用函数，绘制图像
        fig = plot_embedding(result, label, tittle)
        fig.savefig(os.path.join(save_path, tittle + '.png'), dpi=400)
        # 显示图像
        plt.show()

def get_shape(labels):
    # shape_list = ['■', '▲', '♦', '▰', '♥', '⬣', '●', '♣', '♠', '♪']
    shape_list = ['●', '●', '●', '●', '●', '●', '●', '●', '●', '●', '●', '●', '●', '●', '●']
    # shape_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
    label_list = []
    for i in range(len(labels)):
        label_list.append(shape_list[labels[i]])
    return label_list

def plot_embedding(data, label, title):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)     # 对数据进行归一化处理
    fig = plt.figure()      # 创建图形实例
    ax = plt.subplot(111)       # 创建子图
    labels = get_shape(label)
    # 遍历所有样本
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签△
        if data[i, 0] < 0 or data[i, 0] > 1 or data[i, 1] < 0 or data[i, 1] > 1:
            continue
        if label[i] == 1:
            temp_color = plt.cm.Set3(10)
        else:
            temp_color = plt.cm.Set3(label[i])

        plt.text(data[i, 0], data[i, 1], labels[i], color=temp_color,
                 fontdict={'size': 10})

    plt.xticks([-0.1, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1])
    plt.yticks([-0.1, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1])
    frame = plt.gca()
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.get_xaxis().set_visible(False)
    plt.title(title, fontsize=14)

    return fig

# 主函数
if __name__ == '__main__':
    pass
