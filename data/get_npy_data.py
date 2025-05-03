import numpy as np
from collections import defaultdict
import os


def _to_npy_data(dataset_name):
    """
    .txt文件存储为.npy格式，并得到一些统计信息
    """
    dataset_root_path = '../datasets/{}/'.format(dataset_name)
    train_txt_filepath = dataset_root_path + 'train.txt'
    test_txt_filepath = dataset_root_path + 'test.txt'
    train_npy_filepath = dataset_root_path + 'train_data.npy'
    test_npy_filepath = dataset_root_path + 'test_data.npy'

    if os.path.exists(train_txt_filepath):
        train_data = np.loadtxt(fname=train_txt_filepath, dtype=np.int64)
        _, num_columns = train_data.shape
        if num_columns == 3:
            train_user_ids, train_item_ids, _ = train_data.T  # recommendation with implicit data
        else:
            train_user_ids, train_item_ids = train_data.T
        train_data = defaultdict(list)  # maybe it's conducive to get statistics information about data or to save storage?
        for user_id, item_id in zip(train_user_ids, train_item_ids):
            train_data[user_id].append(item_id)
        np.save(train_npy_filepath, train_data)
    else:
        train_user_ids, train_item_ids = [], []

    if os.path.exists(test_txt_filepath):
        test_data = np.loadtxt(fname=test_txt_filepath, dtype=np.int64)
        _, num_columns = test_data.shape
        if num_columns == 3:
            test_user_ids, test_item_ids, _ = test_data.T
        else:
            test_user_ids, test_item_ids = test_data.T
        test_data = defaultdict(list)
        for user_id, item_id in zip(test_user_ids, test_item_ids):
            # if user_id not in test_data:
            #     test_data[user_id] = []
            test_data[user_id].append(item_id)
        np.save(test_npy_filepath, test_data)
    else:
        test_user_ids, test_item_ids = [], []

    # 训练集中user和item数量
    train_dataset_length = len(list(train_user_ids))
    train_user_num = len(set(list(train_user_ids)))
    train_item_num = len(set(list(train_item_ids)))
    print('The length of train set is : \033[33m {} \033[0m'.format(train_dataset_length))
    print('The number of users and items in train set are \033[33m {} \033[0m,\033[33m {} \033[0m respectively'
          .format(train_user_num, train_item_num))
    # 测试集中user和item数量
    test_dataset_length = len(list(test_user_ids))
    test_user_num = len(set(list(test_user_ids)))
    test_item_num = len(set(list(test_item_ids)))
    print('The length of test set is : \033[33m {} \033[0m'.format(test_dataset_length))
    print('The number of users and items in test set are \033[33m {} \033[0m,\033[33m {} \033[0m respectively'
          .format(test_user_num, test_item_num))
    # 总的user和item
    user_num = len(set(list(train_user_ids) + list(test_user_ids)))
    item_num = len(set(list(train_item_ids) + list(test_item_ids)))
    print('The number of users and items are \033[33m {} \033[0m,\033[33m {} \033[0m respectively'.format(user_num, item_num))
    # 构建矩阵时应该使用的user和item数量
    user_id_max = max(max(train_user_ids), max(test_user_ids))
    item_id_max = max(max(train_item_ids), max(test_item_ids))
    print('The shape of user-item interaction matrix should be : \033[34m {} \033[0m, '
          '\033[34m {} \033[0m'.format(user_id_max+1, item_id_max+1))


if __name__ == '__main__':
    # 更换数据集名字
    _to_npy_data('amazon-book')

