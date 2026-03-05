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

import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs


        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay
        self.layer_vectors = {}
        self.weight_vectors = {}
        self.layer_left_subspaces = {}


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))

    def make_vectors(self):
        # for name, layer in self.model.named_children():
        #     params = [param.data.flatten() for param in layer.parameters()]
        #     layer_vector = torch.cat(params)
        #     self.layer_vectors[name] = layer_vector
        for name, layer in self.model.named_children():
            # 如果该层有可学习参数
            if list(layer.parameters()):
                params = [param.data.flatten() for param in layer.parameters()]
                layer_vector = torch.cat(params)
                self.layer_vectors[name] = layer_vector

    def SVD_SPLIT(self, energy_thresh=0.95):
        # 对每一层进行 SVD 分解，按能量阈值选择秩 k，并收集左奇异向量子空间
        self.layer_left_subspaces = {}  # 存放每层的左子空间列表
        for name, layer in self.model.named_children():
            if not list(layer.parameters()):
                continue
            left_subspaces = []  # 存放当前层每个参数块的左子空间
            for param in layer.parameters():
                p = param.data.clone().detach().to(self.device)
                # 将参数视作二维矩阵：如果是 2D 或更高维，保留第一个维度为行，其余展平为列
                if p.dim() >= 2:
                    matrix = p.view(p.shape[0], -1)
                else:
                    # 对于一维向量（比如 bias），把它视为 1 x N 的矩阵
                    matrix = p.view(1, -1)
                # 计算 SVD（兼容不同 PyTorch 版本）
                try:
                    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
                except AttributeError:
                    # 旧版 torch.svd 返回 U, S, V
                    U, S, V = torch.svd(matrix)
                    Vh = V.t()
                # 计算奇异值能量占比并选择最小 k 使得累计能量 >= energy_thresh
                energy = S.pow(2)
                total_energy = energy.sum()
                if total_energy == 0:
                    k = 1
                else:
                    cum_energy = torch.cumsum(energy, dim=0) / total_energy
                    idx = torch.nonzero(cum_energy >= energy_thresh, as_tuple=False)
                    k = int(idx[0].item() + 1) if idx.numel() > 0 else S.numel()
                # 取前 k 列作为左子空间基
                U_reduced = U[:, :k].contiguous()
                left_subspaces.append(U_reduced)
            # 保存该层的左子空间列表
            self.layer_left_subspaces[name] = left_subspaces

    def visual(self):
        train_loader = self.load_train_data()
        X = []
        y = []

        for data, label in train_loader:
            X.extend(data.numpy())
            y.extend(label.numpy())

        X = np.array(X).reshape(len(X), -1)
        y = np.array(y)
        tsne = TSNE(n_components=2, random_state=0)
        X_2d = tsne.fit_transform(X)

        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=50)
        scatter.set_clim(0, 9)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Classes', fontsize=20)
        plt.title(f'Client {self.id} Data Distribution (t-SNE)', fontsize=20)
        plt.axis('off')  # 去除坐标轴和边框
        plt.show()
