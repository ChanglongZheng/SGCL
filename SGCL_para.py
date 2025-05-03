from utils.tools import ensure_dir


def SGCL_parameters(run_id, dataset_name):
    data_path = './datasets/{}/'.format(dataset_name)
    log_save_path = './saved/dataset_' + dataset_name + '/model_SGCL/' + 'run_id_' + str(run_id) + '/'
    model_save_path = log_save_path + 'models_save/'
    ensure_dir(model_save_path)
    print('model save path is : ', model_save_path)
    data_config = {
        'data_path': data_path,
        'log_save_path': log_save_path,
        'model_save_path': model_save_path,
        'dataset_name': dataset_name,
        'topks': 20,
        'lr': 0.001,
        'num_negative': 1,
        'early_stops': 20,
        'seed': 2025,
    }

    if dataset_name == 'douban-book':
        data_config['latent_dim'] = 64
        data_config['num_user'] = 13024
        data_config['num_item'] = 22347
        data_config['gcn_layer'] = 2
        data_config['epochs'] = 200
        data_config['batch_size'] = 2048
        data_config['l2_reg'] = 1e-4 # L2正则项系数
        data_config['ssl_reg_alpha'] = 0.2 # 第一个对比学习loss的权重
        data_config['ssl_reg_beta'] = 0.5 # 第二个对比学习loss的权重
        data_config['ssl_temp'] = 0.2  # 温度系数
        data_config['epsilon'] = 0.1 # 噪声大小
    elif dataset_name == 'yelp2018':
        data_config['latent_dim'] = 64
        data_config['num_user'] = 31668
        data_config['num_item'] = 38048
        data_config['gcn_layer'] = 2
        data_config['epochs'] = 200
        data_config['batch_size'] = 2048
        data_config['l2_reg'] = 1e-4
        data_config['ssl_reg_alpha'] = 0.2
        data_config['ssl_reg_beta'] = 0.5
        data_config['ssl_temp'] = 0.15
        data_config['epsilon'] = 0.1
    elif dataset_name == 'amazon-kindle':
        data_config['latent_dim'] = 64
        data_config['num_user'] = 139823
        data_config['num_item'] = 98824
        data_config['gcn_layer'] = 2
        data_config['epochs'] = 200
        data_config['batch_size'] = 2048
        data_config['l2_reg'] = 1e-4
        data_config['ssl_reg_alpha'] = 0.2
        data_config['ssl_reg_beta'] = 0.1
        data_config['ssl_temp'] = 0.2
        data_config['epsilon'] = 0.1
    elif dataset_name == 'amazon-book':
        data_config['latent_dim'] = 64
        data_config['num_user'] = 52643
        data_config['num_item'] = 91599
        data_config['gcn_layer'] = 2
        data_config['epochs'] = 200
        data_config['batch_size'] = 2048
        data_config['l2_reg'] = 1e-4
        data_config['ssl_reg_alpha'] = 2.0
        data_config['ssl_reg_beta'] = 1.0
        data_config['ssl_temp'] = 0.2
        data_config['epsilon'] = 0.1

    else:
        raise ValueError('Can\'t find the parameters setting for {}'.format(dataset_name))

    return data_config


