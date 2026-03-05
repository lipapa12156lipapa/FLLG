# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import torch
import numpy as np
import h5py
import copy
import time
import random
from utils.data_utils import read_client_data
from utils.dlg import DLG
import os
from datetime import datetime
import json
import torch.nn.functional as F

class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = {}
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        self.algo = args.algorithm

        self.Budget = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

        self.prev_uploaded_weights = {}
        self.step_size = 0.1
        self.avg_loss = 0
        self.processed_indices = set()
        self.cut_clients = []
        self.cut_clients_ids = []
        self.layer_vectors = {}
        self.weight_vectors = {}
        self.layer_left_subspaces = {}  # 存放每层的左子空间列表



    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1),
                                                             1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models_FLLG(self, i):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)
        self.aggregate_weights(i, self.step_size, metric="l2")
        self.weight_vectors.clear()
        for client in self.selected_clients:
            for name, weight in client.weight_vectors.items():
                if name not in self.weight_vectors:
                    self.weight_vectors[name] = []
                self.weight_vectors[name].append(weight)

    def receive_models(self, i):

        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    # 将每一层的参数拼接成一个大的向量
    def make_vector(self):#每层的参数拼接成一个大向量，存在self.layer_vectors里面
        for name, layer in self.global_model.named_children():
            if list(layer.parameters()):
                params = [param.data.flatten() for param in layer.parameters()]
                layer_vector = torch.cat(params)
                self.layer_vectors[name] = layer_vector

    # def aggregate_weights(self, i, step_size=0.1):
    #     if i == 0:
    #         self.make_vector()
    #         tot_samples = 0
    #         for client in self.selected_clients:
    #             tot_samples += client.train_samples
    #         for name, weight in self.global_model.named_children():
    #             for client in self.selected_clients:
    #                 client.weight_vectors[name] = client.train_samples/tot_samples
    #
    #     else:
    #         l2_norms = {}
    #         mid_weight = {}
    #         for name, global_layer_vector in self.layer_vectors.items():
    #             l2_norms[name] = []
    #             mid_weight[name] = 0
    #             for client in self.selected_clients:
    #                 client_vector = client.layer_vectors[name]
    #                 l2_norm = torch.norm(global_layer_vector - client_vector, p=2)
    #                 l2_norms[name].append(l2_norm.item())#计算差异
    #             for l2_norm, client in zip(l2_norms[name], self.selected_clients):
    #                 client.weight_vectors[name] += (l2_norm - (sum(l2_norms[name])/len(l2_norms[name]))
    #                                                 * step_size / max(l2_norms[name]))#动态聚合
    #                 mid_weight[name] += client.weight_vectors[name]#中间值
    #
    #             for client in self.selected_clients:
    #                 client.weight_vectors[name] = client.weight_vectors[name] / mid_weight[name]

    def aggregate_weights(self, i, step_size=0.1, metric="cos", eps=1e-12):
        if i == 0:
            self.make_vector()
            tot_samples = 0
            for client in self.selected_clients:
                tot_samples += client.train_samples
            for name, weight in self.global_model.named_children():
                for client in self.selected_clients:
                    client.weight_vectors[name] = client.train_samples / tot_samples

        else:
            l2_norms = {}
            mid_weight = {}
            for name, global_layer_vector in self.layer_vectors.items():
                l2_norms[name] = []
                mid_weight[name] = 0
                for client in self.selected_clients:
                    client_vector = client.layer_vectors[name]

                    # ======= ONLY CHANGE: gap metric =======
                    if metric == "l2":
                        l2_norm = torch.norm(global_layer_vector - client_vector, p=2)
                    elif metric == "l1":
                        l2_norm = torch.norm(global_layer_vector - client_vector, p=1)
                    elif metric in ["cos", "cosine"]:
                        sim = F.cosine_similarity(
                            global_layer_vector.view(-1).float(),
                            client_vector.view(-1).float(),
                            dim=0,
                            eps=eps
                        )  # [-1, 1]
                        l2_norm = 1.0 - sim  # cosine distance in [0, 2]
                    else:
                        raise ValueError(f"Unknown metric: {metric}")
                    # ======================================

                    l2_norms[name].append(l2_norm.item())  # 计算差异

                for l2_norm, client in zip(l2_norms[name], self.selected_clients):
                    client.weight_vectors[name] += (l2_norm - (sum(l2_norms[name]) / len(l2_norms[name]))
                                                    * step_size / max(l2_norms[name]))  # 动态聚合
                    mid_weight[name] += client.weight_vectors[name]  # 中间值

                for client in self.selected_clients:
                    client.weight_vectors[name] = client.weight_vectors[name] / mid_weight[name]

    def aggregate_parameters_FLLG(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])

        for name, weight_list in self.weight_vectors.items():
            global_layer = dict(self.global_model.named_children())[name]

            for global_param in global_layer.parameters():
                global_param.data.zero_()

            for i, weight in enumerate(weight_list):
                local_layer = dict(self.uploaded_models[i].named_children())[name]
                for global_param, client_param in zip(global_layer.parameters(), local_layer.parameters()):
                    global_param.data += client_param.data.clone() * weight

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        self.rs_test_auc.append(test_auc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc

    def output(self):
        if not os.path.exists("output"):
            os.makedirs("output")

        output_dir = f"output/output_{self.algo}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = f"{output_dir}/output_{self.dataset}.txt"

        mode = "a" if os.path.exists(output_file) else "w"
        with open(output_file, "a", encoding="utf-8") as file:
            file.write(f"本轮参数为数据集：{self.args.dataset}，"
                       f"全局轮数：{self.global_rounds}，客户端数量：{self.num_clients}，"
                       f"学习率为：{self.learning_rate}，时间：{datetime.now()}，"
                       f"模型为：{self.args.model}\n，运行时间大约为：{self.Budget[-1]:.2f}秒\n")

            file.write("test_acc\n")
            for index, value in enumerate(self.rs_test_acc):
                file.write(f'{value}\n')
            # for index, value in enumerate(self.rs_test_acc):
            #     if index % 2 == 0:
            #         file.write(f'{value}\n')
            #
            # file.write("ptrain:\n")
            #
            # for index, value in enumerate(self.rs_test_acc):
            #     if index % 2 == 1:
            #         file.write(f'{value}\n')

            file.write("test_auc\n")
            for index, value in enumerate(self.rs_test_auc):
                file.write(f'{value}\n')

            file.write("avg_loss\n")
            for index, value in enumerate(self.rs_train_loss):
                file.write(f'{value}\n')
            # for index, value in enumerate(self.rs_train_loss):
            #     if index % 2 == 0:
            #         file.write(f'{value}\n')
            #
            # file.write("ptrain:\n")
            #
            # for index, value in enumerate(self.rs_train_loss):
            #     if index % 2 == 1:
            #         file.write(f'{value}\n')

    # 默认这个cut_clients里面不为空
    def concatenation_vector(self):
        for client in self.cut_clients:
            for name, param in client.model.named_parameters():
                client.param_grad[name] = param.grad.clone()

            # 将client里的参数的梯度拼接成一个大向量，存在client.grad_vector里面
            client.grad_vector = torch.cat([grad.flatten() for grad in client.param_grad.values()])

        # 创建一个余弦相似度矩阵
        cos_sim_matrix = torch.zeros(len(self.cut_clients), len(self.cut_clients))

        # 计算两两之间的余弦相似度
        for i in range(len(self.cut_clients)):
            model1_grad_vector = self.cut_clients[i].grad_vector
            for j in range(i+1, len(self.cut_clients)):
                model2_grad_vector = self.cut_clients[j].grad_vector
                cos_sim = torch.cosine_similarity(model1_grad_vector, model2_grad_vector, dim=0)
                cos_sim_matrix[i, j] = cos_sim
                cos_sim_matrix[j, i] = cos_sim


        print(cos_sim_matrix)

        for index, client in enumerate(self.cut_clients):
            print(self.cut_clients[index].id, client.aggregate_weight)
