import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import numba as nb
import random
from time import time
from collections import defaultdict
import torch.nn.functional as F

@nb.njit()
def negative_sampling(train_data_numba, num_item, num_negative=1):
    """
    使用numba加速采样
    :param train_data_numba: 以nb.typed.Dict形式存储训练集
    :param num_item: item数量
    :param num_negative: bpr负样本数量，默认为1
    :return: [user, positive sample, negative sample]三元组
    """
    train_data_triplet = []
    for user in train_data_numba:
        for pos_i in train_data_numba[user]:
            for _ in range(num_negative):
                neg_i = random.randint(0, num_item - 1)
                while neg_i in train_data_numba[user]:
                    neg_i = random.randint(0, num_item - 1)
                train_data_triplet.append([user, pos_i, neg_i])
    return np.array(train_data_triplet)


class BaseCF(nn.Module):
    def __init__(self, data_config):
        super(BaseCF, self).__init__()
        self.data_path = data_config['data_path']
        self.batch_size = data_config['batch_size']
        self.num_user = data_config['num_user']
        self.num_item = data_config['num_item']
        self.num_node = self.num_user + self.num_item
        self.latent_dim = data_config['latent_dim']
        self.l2_reg = data_config['l2_reg']
        self.lr = data_config['lr']
        self.seed = data_config['seed']

        # 得到defaultdict(list)的训练集和测试集
        self.train_data = np.load(self.data_path + 'train_data.npy', allow_pickle=True).tolist()  # convert to defaultdict(list)
        self.test_data_original = np.load(self.data_path + 'test_data.npy', allow_pickle=True).tolist()  # convert to defaultdict(list)

        # 原始数据中的user column和item column
        self.training_user, self.training_item = [], []
        for user, items in self.train_data.items():
            self.training_user.extend([user] * len(items))
            self.training_item.extend(items)

        # 原始测试集中的user column和item column
        self.testing_user, self.testing_item = [], []
        for user, items in self.test_data_original.items():
            self.testing_user.extend([user] * len(items))
            self.testing_item.extend(items)

        '''
        处理异常数据，在测试集中存在一些没有在训练集中出现的用户和物品（在douban-book中会有这种情况，其他数据集没发现）
        '''
        self.test_data = defaultdict(list)
        # 处理开始
        user_set = set(self.training_user)
        item_set = set(self.training_item)
        for user, item in list(zip(self.testing_user, self.testing_item)):
            if user not in user_set or item not in item_set:
                continue
            self.test_data[user].append(item)
        # 处理结束

        # 将数据格式转化为numba类型
        self.train_data_numba = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.int64[:])
        for key, values in self.train_data.items():
            if len(values) > 0:
                self.train_data_numba[key] = np.asarray(list(values))

        # embedding初始化，统一使用xa_uniform初始化
        self.user_embedding = nn.Embedding(num_embeddings=self.num_user, embedding_dim=self.latent_dim, dtype=torch.float32)
        self.item_embedding = nn.Embedding(num_embeddings=self.num_item, embedding_dim=self.latent_dim, dtype=torch.float32)
        self._weight_init(init_type='xa_uniform')

        self.trainDataSize = len(self.training_user)

    def _weight_init(self, init_type='xa_norm'):
        if init_type == 'norm':
            nn.init.normal_(self.user_embedding.weight, std=0.1)
            nn.init.normal_(self.item_embedding.weight, std=0.1)
        elif init_type == 'xa_norm':
            nn.init.xavier_normal_(self.user_embedding.weight)
            nn.init.xavier_normal_(self.item_embedding.weight)
        elif init_type == 'xa_uniform':
            nn.init.xavier_uniform_(self.user_embedding.weight)
            nn.init.xavier_uniform_(self.item_embedding.weight)
        else:
            print('Add others initialize ways.')

    def create_adj_mat(self):
        user_np = np.array(self.training_user)
        item_np = np.array(self.training_item)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_user)), shape=(self.num_node, self.num_node))
        adj_mat = tmp_adj + tmp_adj.T
        return adj_mat

    @staticmethod
    def norm_adj_mat(adj_mat):
        # normalize way: D^-1/2 * A * D^-1/2
        row_sum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(row_sum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)  # 得到度数矩阵D^-1/2
        norm_adj = d_mat_inv.dot(adj_mat.dot(d_mat_inv))
        return norm_adj

    def batch_sampling(self, num_negative):
        """
        Sample batches for an epoch.
        :param num_negative: bpr负样本数量，默认为1
        :return: an iter used to take out data of every batch
        """
        t1 = time()
        triplet_data = negative_sampling(self.train_data_numba, self.num_item, num_negative)
        print('\nPrepare triplet format data cost: {:.2f}'.format(time() - t1))
        batch_num = len(triplet_data) // self.batch_size + 1  # 确定batch的数量
        # 在每个epoch中都进行一次打乱，在LightGCN-pytorch版本中也有这一步操作
        indices = np.arange(triplet_data.shape[0])
        np.random.shuffle(indices)
        # get minibatch
        for k in range(batch_num):
            index_start = k * self.batch_size
            index_end = min((k + 1) * self.batch_size, len(indices))
            # 这里可以直接返回剩余的数据，不凑够一个batch；也可以去凑一个batch的数据；LightGCN选择前者，但这两种在结果上没什么区别
            if index_end == len(indices):
                index_start = len(indices) - self.batch_size
            batch_data = triplet_data[indices[index_start:index_end]]
            yield batch_data[:, 0], batch_data[:, 1], batch_data[:, 2]

    @staticmethod
    def get_embedding(user_emb, item_emb, users_id, pos_items_id, neg_items_id):
        u_emb = user_emb[users_id]
        pos_emb = item_emb[pos_items_id]
        neg_emb = item_emb[neg_items_id]
        return u_emb, pos_emb, neg_emb

    @staticmethod
    def compute_bpr_loss(user_emb, pos_emb, neg_emb):
        pos_scores = torch.sum(torch.mul(user_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(user_emb, neg_emb), dim=1)
        # 按照LightGCN中的做法换成softplus实现sigmoid操作
        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        auc = torch.mean((pos_scores > neg_scores).float())
        return bpr_loss, auc

    # 使用0层的embedding计算L2正则化损失
    def compute_reg_loss_LGN(self, user_idx, pos_item_idx, neg_item_idx):
        user_emb_ego, pos_item_emb_ego, neg_item_emb_ego = (self.user_embedding(user_idx),
                                                            self.item_embedding(pos_item_idx), self.item_embedding(neg_item_idx))
        reg_loss = (1 / 2) * (user_emb_ego.norm(2).pow(2) + pos_item_emb_ego.norm(2).pow(2) +
                              neg_item_emb_ego.norm(2).pow(2)) / float(len(user_idx))
        reg_loss = self.l2_reg * reg_loss
        return reg_loss

    @staticmethod
    def convert_csr_to_spare_tensor(csr_input: sp.csr_matrix) -> torch.sparse_coo_tensor:
        coo = csr_input.tocoo()
        indices = torch.tensor(np.mat([coo.row, coo.col])).long()
        data = torch.tensor(coo.data, dtype=torch.float32)  # force the datatype as torch.float32
        shape = torch.Size(coo.shape)
        return torch.sparse_coo_tensor(indices, data, shape)

    def edge_dropout_adj(self, drop_ratio):
        """ adjacent matrix after edge dropout """
        index = np.arange(len(self.training_user))
        keep_index = np.random.choice(index, size=int(len(self.training_user) * (1 - drop_ratio)), replace=False)
        user_np = np.array(self.training_user)[keep_index]
        item_np = np.array(self.training_item)[keep_index]
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_user)), shape=(self.num_node, self.num_node))
        adj_mat = tmp_adj + tmp_adj.T
        norm_adj = self.norm_adj_mat(adj_mat)
        return norm_adj


