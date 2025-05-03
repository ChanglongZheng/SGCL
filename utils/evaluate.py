import faiss
import numpy as np
import math
from collections import defaultdict
import numba as nb


@nb.njit()
def compute_ranking_metrics(testset_users, trainset_items, testset_items, top_k_list, user_ranked_item_list):
    """ return: the hit_ratio@n, recall@n, ndcg@n for all users in the test dataset """
    all_metrics = []
    for i in nb.prange(len(testset_users)):
        user = testset_users[i]
        one_metrics = []
        mask_items = trainset_items[i]  # the items in train data need to be masked
        test_items = testset_items[i]  # a specific user's ground truth label
        test_items_length = len(test_items)

        # filter out the real predication items, just get the longest prediction item list for more top_ks metrics computation
        pred_items_all = user_ranked_item_list[user]
        max_length_candidate = len(mask_items) + top_k_list[-1]
        pred_items = [item for item in pred_items_all[:max_length_candidate] if item not in mask_items][:top_k_list[-1]]

        # metrics computation for top_ks
        for top_k in top_k_list:
            # compute hits and dcg
            hit_value = 0
            dcg_value = 0
            for idx in nb.prange(top_k):
                if pred_items[idx] in test_items:
                    hit_value += 1
                    dcg_value += math.log(2) / math.log(idx+2)
            # compute idcg
            target_length = min(top_k, test_items_length)  # 有些人交互太少了，可能还没有top-k多
            idcg = 0.0
            for k in nb.prange(target_length):
                idcg += math.log(2) / math.log((k+2))

            hit_ratio = hit_value / target_length
            recall = hit_value / test_items_length
            ndcg = dcg_value / idcg
            one_metrics.append([hit_ratio, recall, ndcg])

        all_metrics.append(one_metrics)

    return all_metrics


def evaluate_with_faiss(_train_data, _test_data, _top_k_list, _user_emb, _item_emb):
    """
    :param _train_data: type<defaultdict(list)>
    :param _test_data: type<defaultdict(list)>
    :param _top_k_list: list contains top_ks
    :param _user_emb: user embedding
    :param _item_emb: item embedding
    :return:
    """
    trainset_users = list(_train_data.keys())
    testset_users = list(_test_data.keys())

    # ********** faiss search ***********
    # 把用户emb作为query vectors，把物品emb作为vectors_database
    # 对每个用户，在所有的item中找到内积分数最大的k个items
    query_vectors = _user_emb
    dim = _user_emb.shape[-1]
    index = faiss.IndexFlatIP(dim)  # 使用内积的方式进行搜索
    index.add(_item_emb)  # 加入数据，使用index.ntotal查看加入了多少行数据

    # 输出结果按照分数降序排序
    # 因为是对所有的item做搜索，还去除训练集中的items
    max_mask_items_length = max(len(_train_data[user]) for user in _train_data.keys())
    inner_product_scores, _user_ranked_item_list = index.search(query_vectors, _top_k_list[-1] + max_mask_items_length)
    trainset_items = [list(_train_data[user]) if user in trainset_users else [-1] for user in testset_users]
    testset_items = [list(_test_data[user]) for user in testset_users]

    trainset_items_numba = nb.typed.List()
    for i in range(len(trainset_items)):
        trainset_items_numba.append(nb.typed.List(trainset_items[i]))
    testset_items_numba = nb.typed.List()
    for i in range(len(testset_items)):
        testset_items_numba.append(nb.typed.List(testset_items[i]))

    all_metrics = compute_ranking_metrics(testset_users=nb.typed.List(testset_users),
                                          trainset_items=trainset_items_numba,
                                          testset_items=testset_items_numba,
                                          top_k_list=nb.typed.List(_top_k_list),
                                          user_ranked_item_list=nb.typed.List(_user_ranked_item_list))

    # in case of many top_ks
    hr_top_k_list = defaultdict(list)
    recall_top_k_list = defaultdict(list)
    ndcg_top_k_list = defaultdict(list)
    hr_mean, recall_mean, ndcg_mean = {}, {}, {}

    # rearrange the metrics
    for i, one_metrics in enumerate(all_metrics):
        j = 0
        for top_k in _top_k_list:
            hr_top_k_list[top_k].append(one_metrics[j][0])
            recall_top_k_list[top_k].append(one_metrics[j][1])
            ndcg_top_k_list[top_k].append(one_metrics[j][2])
            j += 1

    for top_k in _top_k_list:
        hr_mean[top_k] = np.mean(hr_top_k_list[top_k])
        recall_mean[top_k] = np.mean(recall_top_k_list[top_k])
        ndcg_mean[top_k] = np.mean(ndcg_top_k_list[top_k])

    return hr_mean, recall_mean, ndcg_mean


