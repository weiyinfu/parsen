from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

"""
研一下学期的代码，2017年上半年

那时候numpy都用的不熟
"""


def kernel_uniform(u, h, dim):  # 矩形窗核函数，均匀分布
    return 1 / (h ** dim) if u <= 1 / 2 else 0


def kernel_gauss(u, h):  # 高斯核函数，mu和sigma
    return np.exp(-u * u / (2 * h * h)) / (np.sqrt(2 * np.pi) * h)


def norm_max(vec):  # 向量的无穷范数
    return np.max(np.abs(vec))


def norm2(vec: np.ndarray):  # 向量的2范数
    return np.linalg.norm(vec)


def load_letter_recognition(test_size: float):
    # test_size：测试集的比例
    data = pd.read_csv("letter-recognition.data").values
    train_data, test_data = train_test_split(data, test_size=test_size)
    a = defaultdict(lambda: [])  # 把测试数据按照类别分开
    for i in train_data:
        a[i[0]].append(i[1:])
    return a, test_data


def get_ans(test_sample, train_data, h, kernel, norm):
    ans = None  # 目标类别
    ansP = 0  # 目标类别的概率
    test_vec = test_sample[1:]
    for category, samples in train_data.items():  # j中的数据都是同一类别
        s = sum([kernel(norm(sample - test_vec), h) for sample in samples])
        if s > ansP:
            ansP = s
            ans = category
    return ans


def run():
    train_data, test_data = load_letter_recognition(test_size=0.2)
    print('训练集', train_data.keys(), test_data.shape)
    right_cnt = 0  # 正确的个数
    h = 3  # 窗口大小，窗口大小拍脑袋决定不太好
    for i in tqdm(test_data):
        # ans = get_ans(i, train_data, h, kernel_uniform, norm_max)
        ans = get_ans(i, train_data, h, kernel_gauss, norm2)
        if ans == i[0]:
            right_cnt += 1
    print('正确率', right_cnt / len(test_data))


if __name__ == '__main__':
    run()
