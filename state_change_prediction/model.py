from __future__ import division
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.optim as optim
import math, random
import torch
from collections import defaultdict
import os
PATH = "./"
total_reinitialization_count = 0


class MLP(nn.Module):
    def __init__(self, mlp_num_layers, in_dim, out_dim):
        super(MLP, self).__init__()
        self.linear_or_not = True
        self.mlp_num_layers = mlp_num_layers
        self.out_dim = out_dim

        if self.mlp_num_layers == 1:
            self.linear = nn.Linear(in_dim, out_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.linears.append(nn.Linear(in_dim, out_dim))
            for layer in range(mlp_num_layers - 2):
                self.linears.append(nn.Linear(out_dim, out_dim))
            self.linears.append(nn.Linear(out_dim, out_dim))

    def forward(self, x):
        if self.linear_or_not:
            x = self.linear(x)
            return x
        else:
            h = x
            for i in range(self.mlp_num_layers-1):
                h = self.linears[i](h)
                h = F.relu(h)
            h=self.linears[-1](h)
            return h


class NeuralKF(nn.Module):
    def __init__(self, num_mlp_layers, num_inst, emb_dim, num_features, device):
        super(NeuralKF, self).__init__()
        self.num_inst = num_inst
        self.emb_dim = emb_dim
        self.device = device
        self.no_feature = False
        if num_features == 0:
            self.no_feature = True

        if not self.no_feature:
            self.tran_mlp = MLP(num_mlp_layers, emb_dim + num_features, emb_dim)
        self.pred_mlp = MLP(num_mlp_layers, emb_dim, emb_dim)
        self.K_mlp_in = MLP(num_mlp_layers, self.emb_dim + 1, self.emb_dim)
        self.K_gru = nn.GRUCell(self.emb_dim, self.emb_dim)
        self.h = torch.zeros((num_inst, self.emb_dim)).to(self.device)
        self.K_mlp_out = MLP(num_mlp_layers, self.emb_dim, self.emb_dim)

    def forward(self, ids, emb_prio, emb, dual, feat):
        if not self.no_feature:
            emb = self.tran_mlp(torch.cat([emb, feat], dim=1))
        emb_post_pred = self.pred_mlp(emb)
        emb_res = emb_post_pred - emb_prio
        z_res = torch.ones((emb.shape[0], 1)).to(self.device) - torch.sum(torch.mul(dual, emb_prio), dim=1, keepdim=True)

        self.K_in = self.K_mlp_in(torch.cat([emb_res, z_res], dim=1))
        h_state = self.h[ids, :]
        h_state = self.K_gru(self.K_in, h_state)
        self.h[ids, :] = h_state
        self.K = self.K_mlp_out(h_state)

        emb_post = emb_prio + self.K * z_res
        emb_regu = torch.norm(emb_post_pred - emb_post)
        return emb_post, emb_regu


class NeuFilter(nn.Module):
    def __init__(self, args, num_features, num_users, num_items, device):
        super(NeuFilter, self).__init__()
        self.embedding_dim = args.embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.num_features = num_features
        self.user_static_embedding_size = num_users
        self.item_static_embedding_size = num_items
        self.device=device

        self.initial_user_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))
        self.initial_item_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))

        self.user_kf = NeuralKF(args.num_layer, num_users, self.embedding_dim, num_features, device)
        self.item_kf = NeuralKF(args.num_layer, num_items, self.embedding_dim, num_features, device)

        self.linear_layer1 = nn.Linear(self.embedding_dim, 50)
        self.linear_layer2 = nn.Linear(50, 2)
        self.prediction_layer = nn.Linear(self.user_static_embedding_size + self.item_static_embedding_size + self.embedding_dim * 2, self.item_static_embedding_size + self.embedding_dim)
        self.pe_div_term = torch.exp(torch.arange(0, self.embedding_dim, 2) * (-(math.log(10000.0) / (self.embedding_dim)))).to(device)

        self.user_project_layer = MLP(args.num_layer, self.embedding_dim, self.embedding_dim)
        self.item_project_layer = MLP(args.num_layer, self.embedding_dim, self.embedding_dim)

    def forward(self, user_embeddings, item_embeddings, user_prior=None, item_prior=None, users=None, items=None, user_timediffs=None, item_timediffs=None, features=None, select=None):
        if select == 'user_update':
            user_embedding_output, user_emb_regu = self.user_kf(users, user_prior, user_embeddings, item_embeddings, features)
            return F.normalize(user_embedding_output), user_emb_regu

        elif select == 'item_update':
            item_embedding_output, item_emb_regu = self.item_kf(items, item_prior, item_embeddings, user_embeddings, features)
            return F.normalize(item_embedding_output), item_emb_regu

        elif select == 'project':
            user_positional_encoding = self.get_positional_encoding(user_timediffs)
            item_positional_encoding = self.get_positional_encoding(item_timediffs)
            user_projected_embedding = self.user_project_layer(user_embeddings+user_positional_encoding)
            item_projected_embedding = self.item_project_layer(item_embeddings+item_positional_encoding)
            return user_projected_embedding, item_projected_embedding

    def get_positional_encoding(self, positions):
        positional_encoding = torch.zeros(positions.shape[0], self.embedding_dim).to(self.device)
        positional_encoding[:, 0::2] = torch.sin(positions * self.pe_div_term)
        positional_encoding[:, 1::2] = torch.cos(positions * self.pe_div_term)
        return positional_encoding

    def predict_label(self, user_embeddings):
        X_out = nn.ReLU()(self.linear_layer1(user_embeddings))
        X_out = self.linear_layer2(X_out)
        return X_out

    def predict_item_embedding(self, user_embeddings):
        X_out = self.prediction_layer(user_embeddings)
        return X_out


def reinitialize_tbatches():
    global current_tbatches_interactionids, current_tbatches_user, current_tbatches_item, current_tbatches_timestamp, current_tbatches_feature, current_tbatches_label, current_tbatches_previous_item
    global tbatchid_user, tbatchid_item, current_tbatches_user_timediffs, current_tbatches_item_timediffs, current_tbatches_user_timediffs_next

    current_tbatches_interactionids = defaultdict(list)
    current_tbatches_user = defaultdict(list)
    current_tbatches_item = defaultdict(list)
    current_tbatches_timestamp = defaultdict(list)
    current_tbatches_feature = defaultdict(list)
    current_tbatches_label = defaultdict(list)
    current_tbatches_previous_item = defaultdict(list)
    current_tbatches_user_timediffs = defaultdict(list)
    current_tbatches_item_timediffs = defaultdict(list)
    current_tbatches_user_timediffs_next = defaultdict(list)

    tbatchid_user = defaultdict(lambda: -1)
    tbatchid_item = defaultdict(lambda: -1)

    global total_reinitialization_count
    total_reinitialization_count += 1


def calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_time_series, y_true, loss_function, device):
    prob = model.predict_label(user_embeddings_time_series[tbatch_interactionids,:])
    y = Variable(torch.LongTensor(y_true).to(device)[tbatch_interactionids])
    loss = loss_function(prob, y)
    return loss


def save_model(model, optimizer, args, epoch, user_embeddings, item_embeddings, train_end_idx, user_embeddings_time_series=None, item_embeddings_time_series=None, path=PATH):
    state = {
            'user_embeddings': user_embeddings.data.cpu().numpy(),
            'item_embeddings': item_embeddings.data.cpu().numpy(),
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'train_end_idx': train_end_idx
    }
    if user_embeddings_time_series is not None:
        state['user_embeddings_time_series'] = user_embeddings_time_series.data.cpu().numpy()
        state['item_embeddings_time_series'] = item_embeddings_time_series.data.cpu().numpy()
    directory = os.path.join(path, 'saved_models/{}'.format(args.postfix))
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, "checkpoint.ep{}.pth.tar".format(epoch))
    torch.save(state, filename)


def load_model(model, optimizer, args, epoch, device):
    filename = PATH + "saved_models/{}/checkpoint.ep{}.pth.tar".format(args.postfix, epoch)
    checkpoint = torch.load(filename)
    args.start_epoch = checkpoint['epoch']
    user_embeddings = Variable(torch.from_numpy(checkpoint['user_embeddings']).to(device))
    item_embeddings = Variable(torch.from_numpy(checkpoint['item_embeddings']).to(device))
    try:
        train_end_idx = checkpoint['train_end_idx'] 
    except KeyError:
        train_end_idx = None

    try:
        user_embeddings_time_series = Variable(torch.from_numpy(checkpoint['user_embeddings_time_series']).to(device))
        item_embeddings_time_series = Variable(torch.from_numpy(checkpoint['item_embeddings_time_series']).to(device))
    except:
        user_embeddings_time_series = None
        item_embeddings_time_series = None

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model = model.to(device)
    return [model, optimizer, user_embeddings, item_embeddings, user_embeddings_time_series, item_embeddings_time_series, train_end_idx]


def set_embeddings_training_end(user_embeddings, item_embeddings, user_embeddings_time_series, item_embeddings_time_series, user_data_id, item_data_id, train_end_idx):
    userid2lastidx = {}
    for cnt, userid in enumerate(user_data_id[:train_end_idx]):
        userid2lastidx[userid] = cnt
    itemid2lastidx = {}
    for cnt, itemid in enumerate(item_data_id[:train_end_idx]):
        itemid2lastidx[itemid] = cnt
    try:
        embedding_dim = user_embeddings_time_series.size(1)
    except:
        embedding_dim = user_embeddings_time_series.shape[1]
    for userid in userid2lastidx:
        user_embeddings[userid, :embedding_dim] = user_embeddings_time_series[userid2lastidx[userid]]
    for itemid in itemid2lastidx:
        item_embeddings[itemid, :embedding_dim] = item_embeddings_time_series[itemid2lastidx[itemid]]
    user_embeddings.detach_()
    item_embeddings.detach_()
