import numpy as np
from collections import defaultdict


# 处理一些并非3列的数据，而是一些sequence排列的数据
def to_npy_data(dataset_name):
    """
    default the original dataset format is .txt, and convert it to .npy format for a faster data loading
    :param dataset_name: the name of the dataset u want to use
    :return: None
    """
    # 数据集路径
    dataset_root_path = '../datasets/{}/'.format(dataset_name)
    train_txt_filepath = dataset_root_path + 'train.txt'
    test_txt_filepath = dataset_root_path + 'test.txt'

    # 按行读取数据，并处理
    train_users, train_items = [], []
    with open(test_txt_filepath, 'r') as file:
        for line in file:
            data = line.split()
            user = [data[0]]
            items = data[1:]
            train_users.extend(user * (len(items)))
            train_items.extend(items)
    with open(test_txt_filepath, 'w') as file:
        for item1, item2 in zip(train_users, train_items):
            file.write(f"{item1} {item2}\n")


if __name__ == '__main__':
    to_npy_data('gowalla')

