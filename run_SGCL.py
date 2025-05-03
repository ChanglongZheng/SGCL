import argparse
import os
import sys
import torch
from tqdm import tqdm
from time import time

from models.SGCL import SGCL
from SGCL_para import SGCL_parameters
from utils.tools import seed_everything, set_color
from utils.log import Logger
from utils.evaluate import evaluate_with_faiss


def parse_args():
    parser = argparse.ArgumentParser(description='SimGCL Parameters')
    parser.add_argument('--run_id', type=int, default=1, help='Experiment ID')
    parser.add_argument('--dataset_name', type=str, default='douban-book', help='Dataset chosen')
    return parser.parse_args()


def train_mode():
    rec_model = SGCL(data_config, device).to(device)
    optimizer = torch.optim.Adam(rec_model.parameters(), lr=data_config['lr'])
    for name, param in rec_model.named_parameters():
        if param.requires_grad:
            log.write(set_color('Parameter: {}, shape: {}\n'.format(name, param.shape), 'yellow'))

    max_hr, max_recall, max_ndcg = 0.0, 0.0, 0.0
    early_stop, best_epoch = 0, 0
    topks = data_config['topks']
    eval_used_topk = 20
    model_files = []
    max_to_keep = 5

    for epoch in tqdm(range(1, data_config['epochs']+1), desc=set_color('Train Processing: ', 'pink'), colour='blue', dynamic_ncols=True):
        t1 = time()  # epoch starts time
        sum_auc, all_bpr_loss, all_ssl_loss, all_reg_loss, all_batch_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        batch_num = 0
        rec_model.train()

        loader = rec_model.batch_sampling(num_negative=data_config['num_negative'])
        for u, i, j in tqdm(loader, desc=f'Training epoch {epoch} : '):
            u = torch.tensor(u, dtype=torch.long).to(device)
            i = torch.tensor(i, dtype=torch.long).to(device)
            j = torch.tensor(j, dtype=torch.long).to(device)

            user_emb, item_emb = rec_model.forward()
            auc, bpr_loss, ssl_loss, reg_loss, batch_loss = rec_model.compute_batch_loss(user_emb, item_emb, u, i, j)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            sum_auc += auc.item()
            all_bpr_loss += bpr_loss.item()
            all_ssl_loss += ssl_loss.item()
            all_reg_loss += reg_loss.item()
            all_batch_loss += batch_loss.item()

            batch_num += 1

        # for each epoch, get the mean loss
        mean_auc = sum_auc / batch_num
        mean_bpr_loss = all_bpr_loss / batch_num
        mean_ssl_loss = all_ssl_loss / batch_num
        mean_reg_loss = all_reg_loss / batch_num
        mean_batch_loss = all_batch_loss / batch_num

        t2 = time()

        log.write(set_color('Epoch:{:d}, Train_AUC:{:.4f}, Loss_BPR:{:.4f}, Loss_cl:{:.4f}, Loss_reg:{:.4f}, Loss_sum:{:.4f}\n'.
                            format(epoch, mean_auc, mean_bpr_loss, mean_ssl_loss, mean_reg_loss ,mean_batch_loss), 'blue'))

        # every five epochs conduct an evaluation
        if epoch % 5 == 0:
            early_stop += 1
            rec_model.eval()
            with torch.no_grad():
                user_emb_eval, item_emb_eval = rec_model.forward()
            user_emb_eval = user_emb_eval.cpu().detach().numpy()
            item_emb_eval = item_emb_eval.cpu().detach().numpy()
            hr, recall, ndcg = evaluate_with_faiss(rec_model.train_data, rec_model.test_data, [topks], user_emb_eval, item_emb_eval)

            # update the best performance
            if recall[20] >= max_recall and ndcg[20] >= max_ndcg:
                best_epoch = epoch
                max_recall = recall[20]
                max_ndcg = ndcg[20]
            log.write(set_color('Current Evaluation: Epoch:{:d}, topk:{:d}, recall:{:.4f}. ndcg:{:.4f}\n'.format(epoch, eval_used_topk, recall[20], ndcg[20]), 'green'))
            log.write(set_color('Best Evaluation: Epoch:{:d}, topk:{:d}, recall:{:.4f}. ndcg:{:.4f}\n'.format(best_epoch, eval_used_topk, max_recall, max_ndcg), 'red'))

            # early stop setting
            if ndcg[20] == max_ndcg:
                early_stop = 0
                best_ckpt = 'epoch_' + str(epoch) + '_ndcg_' + str(ndcg[20]) + '.ckpt'
                file_path = data_config['model_save_path'] + best_ckpt
                torch.save(rec_model.state_dict(), file_path)  # only save the weights of the model
                print('Model has been saved to {}'.format(file_path))

                model_files.append(file_path)
                if len(model_files) > max_to_keep:
                    oldest_file = model_files.pop(0)
                    os.remove(oldest_file)
                    print('Old model {} has been removed'.format(oldest_file))

            t3 = time()
            log.write('train_time:{:.2f}, eval_time:{:.2f}\n\n'.format(t2-t1, t3-t2))

            if epoch > 20 and early_stop > data_config['early_stops']:
                log.write('early stop has been activated at epoch {}\n\n'.format(epoch))
                log.write(set_color('max_recall@20=:{:.4f}, max_ndcg@20=:{:.4f}\n\n'.format(max_recall, max_ndcg), 'green'))
                break

    return file_path


def eval_mode(best_ckpt_path):
    loaded_model = SGCL(data_config, device).to(device)
    loaded_model.load_state_dict(torch.load(best_ckpt_path))
    loaded_model.eval()
    with torch.no_grad():
        user_emb, item_emb = loaded_model.forward()
    user_emb = user_emb.cpu().detach().numpy()
    item_emb = item_emb.cpu().detach().numpy()  # I think we could use faiss-gpu in the future

    hr, recall, ndcg = evaluate_with_faiss(loaded_model.train_data, loaded_model.test_data,
                                           [10, 20, 50, 100], user_emb, item_emb)
    for k in ndcg.keys():
        log.write(set_color('Topk:{:3d}, HR:{:.4f}, Recall:{:.4f}, NDCG:{:.4f}\n'.format(k, hr[k], recall[k], ndcg[k]), 'cyan'))



if __name__ == "__main__":
    # 指定使用的GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    dataset_name = args.dataset_name
    run_id = args.run_id
    data_config = SGCL_parameters(run_id, dataset_name)  # read configuration
    seed_everything(data_config['seed'])
    log = Logger(data_config['log_save_path'])
    log.write(set_color('********** Parameters setting about {} training on {} **********\n'.format('SGCL', dataset_name), 'yellow'))
    for key in data_config.keys():
        log.write(set_color(key + '=' + str(data_config[key]) + '\n', color='yellow'))

    best_model_saved_path = train_mode()

    eval_mode(best_model_saved_path)




