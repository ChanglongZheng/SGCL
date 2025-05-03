import torch
from .BaseCF import BaseCF
import torch.nn.functional as F


class SGCL(BaseCF):
    def __init__(self, data_config, device):
        super().__init__(data_config)
        self.dataset = data_config['dataset_name']
        self.eps = data_config['epsilon'] # 噪声大小
        self.emb_size = data_config['latent_dim']
        self.n_layers = data_config['gcn_layer']
        self.temperature = data_config['ssl_temp'] # 温度系数
        self.ssl_reg_alpha = data_config['ssl_reg_alpha'] # 第一个对比学习loss的权重
        self.ssl_reg_beta = data_config['ssl_reg_beta'] # 第二个对比学习loss的权重
        self.sparse_norm_adj = self.convert_csr_to_spare_tensor(self.norm_adj_mat(self.create_adj_mat())).to(device)
        self.device = device

    def forward(self):
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
        user_rec_embeddings, item_rec_embeddings = torch.split(ego_embeddings, [self.num_user, self.num_item])
        return user_rec_embeddings, item_rec_embeddings

    def adding_random_noise(self, user_emb, item_emb):
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        # SimGCL加噪声的方式
        random_noise = (torch.rand_like(all_emb)).to(self.device)
        noise = self.eps * F.normalize(random_noise, dim=-1) * torch.sign(all_emb)
        all_emb += noise
        use_noised_embeddings, item_noised_embeddings = torch.split(all_emb, [self.num_user, self.num_item])
        return use_noised_embeddings, item_noised_embeddings

    def cl_loss(self, user_emb, item_emb, u_idx, i_idx, j_idx):
        u_idx = torch.Tensor(u_idx).type(torch.long)
        i_idx = torch.Tensor(i_idx).type(torch.long)
        u_idx_unique = torch.unique(u_idx).to(self.device)
        i_idx_unique = torch.unique(i_idx).to(self.device)
        # 两个视图加噪
        user_view_1, item_view_1 = self.adding_random_noise(user_emb, item_emb)
        user_view_2, item_view_2 = self.adding_random_noise(user_emb, item_emb)
        # 选出batch embeddings
        user_view_1_batch, item_view_1_batch = user_view_1[u_idx_unique], item_view_1[i_idx_unique]
        user_view_2_batch, item_view_2_batch = user_view_2[u_idx_unique], item_view_2[i_idx_unique]
        # 用户侧和物品侧的对比学习loss
        user_cl_loss = self.infonce(user_view_1_batch, user_view_2_batch)
        item_cl_loss = self.infonce(item_view_1_batch, item_view_2_batch)
        # 交互数据的对比学习loss，选择其中一个加噪视图来做
        ui_cl_loss = self.infonce_ui(user_view_1[u_idx], item_view_1[i_idx], item_view_1[j_idx])
        ssl_loss = self.ssl_reg_alpha * (user_cl_loss + item_cl_loss) + self.ssl_reg_beta * ui_cl_loss
        return ssl_loss

    def infonce(self, view1, view2):
        view1_norm, view2_norm = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = torch.sum(view1_norm * view2_norm, dim=1)
        pos_score = torch.exp(pos_score / self.temperature)
        # 负样本上的改动
        total_score_unit = torch.matmul(view1_norm, view2_norm.T)
        mask1 = total_score_unit > 0.2
        mask2 = total_score_unit < -0.2
        neg_score = torch.exp(torch.abs(total_score_unit / self.temperature))
        neg_score = neg_score * mask1 + neg_score * mask2
        neg_score = torch.sum(neg_score, dim=1)
        return -torch.mean(torch.log(pos_score / neg_score))

    def infonce_ui(self, u_emb, i_emb, j_emb):
        u_emb, i_emb, j_emb = F.normalize(u_emb, dim=1), F.normalize(i_emb, dim=1), F.normalize(j_emb, dim=1)
        pos_score = torch.sum(u_emb * i_emb, dim=1)
        pos_score = torch.exp(pos_score / self.temperature)
        # 这里可以改动也可以不改，对性能影响不大
        candidate_list = torch.cat([i_emb, j_emb], dim=0)
        neg_score_unit = torch.matmul(u_emb, candidate_list.T)
        neg_score = torch.exp(neg_score_unit / self.temperature)
        neg_score = torch.sum(neg_score, dim=1)
        return -torch.mean(torch.log(pos_score / neg_score))

    def compute_batch_loss(self, user_emb, item_emb, u_idx, i_idx, j_idx):
        u_emb, i_emb, j_emb = self.get_embedding(user_emb, item_emb, u_idx, i_idx, j_idx)
        bpr_loss, auc = self.compute_bpr_loss(u_emb, i_emb, j_emb)
        # 使用的是LightGCN里面的正则项损失函数
        reg_loss = self.compute_reg_loss_LGN(u_idx, i_idx, j_idx)
        cl_loss = self.cl_loss(user_emb, item_emb, u_idx, i_idx, j_idx)
        total_loss = bpr_loss + cl_loss + reg_loss
        return auc, bpr_loss, cl_loss, reg_loss, total_loss

