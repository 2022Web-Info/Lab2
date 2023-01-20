import copy
import os
import random
import collections

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

from data_loader.loader_base import DataLoaderBase


class DataLoader(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size

        kg_data = self.load_kg(self.kg_file)
        self.construct_data(kg_data)
        self.print_info(logging)

        self.laplacian_type = args.laplacian_type
        self.create_adjacency_dict()
        self.create_laplacian_dict()


    def construct_data(self, kg_data):
        '''
            kg_data 为 DataFrame 类型
        '''
        # 1. 为KG添加逆向三元组，即对于KG中任意三元组(h, r, t)，添加逆向三元组 (t, r+n_relations, h)，
        #    并将原三元组和逆向三元组拼接为新的DataFrame，保存在 kg_data 中。

        i_kg_data = copy.deepcopy(kg_data)
        i_kg_data = i_kg_data.rename({'h': 't', 't': 'h'}, axis=1)  # 生成逆向三元组
        i_kg_data['r'] += (max(kg_data['r']) + 1)
        kg_data = pd.concat([kg_data, i_kg_data], axis=0, ignore_index=True)

        # 此处不需要修改，添加两种新关系，即 (user, like, movie) 和 (movie, like_by, user)
        kg_data['r'] += 2

        # 2. 计算关系数，实体数，实体和用户的总数
        self.n_relations = max(kg_data['r']) + 1  # max_index + 1
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1  # max_entity_index + 1
        self.n_users_entities = self.n_users + self.n_entities  # 用户和电影实体数求和

        # 3. 使用 map()函数 将 self.cf_train_data 和 self.cf_test_data 中的 用户索引 范围从[0, num of users)
        #    映射到[num of entities, num of entities + num of users)，并保持原有数据形式和结构不变
        self.cf_train_data = (np.array(list(map(lambda f1: f1 + self.n_entities, self.cf_train_data[0]))), self.cf_train_data[1])  # 二元元组
        self.cf_test_data = (np.array(list(map(lambda f2: f2 + self.n_entities, self.cf_test_data[0]))), self.cf_test_data[1])

        # 4. 将 self.train_user_dict 和 self.test_user_dict 中的用户索引（即key值）范围从[0, num of users)
        #    映射到[num of entities, num of entities + num of users)，并保持原有数据形式和结构不变
        self.train_user_dict = {k1 + self.n_entities: v1 for k1, v1 in self.train_user_dict.items()}  # 遍历user_dict
        self.test_user_dict = {k2 + self.n_entities: v2 for k2, v2 in self.test_user_dict.items()}

        # 5. 以三元组的形式 (user, 0, movie) 重构交互数据，其中 关系0 代表 like
        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[0]  # 覆盖上面的第一列和第三列
        cf2kg_train_data['t'] = self.cf_train_data[1]

        # 6. 以三元组的形式 (movie, 0, user) 重构逆向的交互数据，其中 关系1 代表 like_by
        inverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])  # n*3的全1矩阵
        inverse_cf2kg_train_data['h'] = self.cf_train_data[1]
        inverse_cf2kg_train_data['t'] = self.cf_train_data[0]

        self.kg_train_data = pd.concat([kg_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True)
        self.n_kg_train = len(self.kg_train_data)

        # 7. 根据 self.kg_train_data 构建字典 self.train_kg_dict ，其中key为h, value为tuple(t, r)，
        #    和字典 self.train_relation_dict, 其中key为r，value为tuple(h, t)。
        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)

        for row_ in self.kg_train_data.iterrows():
            h, r, t = row_[1]
            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))



    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))


    def create_adjacency_dict(self):
        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            rows = [e[0] for e in ht_list]
            cols = [e[1] for e in ht_list]
            vals = [1] * len(rows)
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users_entities, self.n_users_entities))
            self.adjacency_dict[r] = adj


    def create_laplacian_dict(self):
        def symmetric_norm_lap(adj):
            # D^{-1/2}AD^{-1/2}
            rowsum = np.array(adj.sum(axis=1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def random_walk_norm_lap(adj):
            # D^{-1}A
            # 8. 根据对称归一化拉普拉斯矩阵的计算代码，补全随机游走归一化拉普拉斯矩阵的计算代码
            rowsum = np.array(adj.sum(axis=1))

            d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)  # D^{-1}A
            return norm_adj.tocoo()

        if self.laplacian_type == 'symmetric':
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == 'random-walk':
            norm_lap_func = random_walk_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            self.laplacian_dict[r] = norm_lap_func(adj)

        A_in = sum(self.laplacian_dict.values())
        self.A_in = self.convert_coo2tensor(A_in.tocoo())


    def print_info(self, logging):
        logging.info('n_users:           %d' % self.n_users)
        logging.info('n_items:           %d' % self.n_items)
        logging.info('n_entities:        %d' % self.n_entities)
        logging.info('n_users_entities:  %d' % self.n_users_entities)
        logging.info('n_relations:       %d' % self.n_relations)

        logging.info('n_cf_train:        %d' % self.n_cf_train)
        logging.info('n_cf_test:         %d' % self.n_cf_test)

        logging.info('n_kg_train:        %d' % self.n_kg_train)


