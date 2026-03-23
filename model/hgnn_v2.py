import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from .backbone.resnet import resnet101
from .semantic import semantic
from .classifier_layer import Classifier_Layer
from copy import deepcopy
import math
from torch.nn.parameter import Parameter
from six.moves import cPickle
import time
import torchvision.models as models


class EmaNetwork(torch.nn.Module):
    #  指数移动平均, 用于平滑模型权重的更新
    def __init__(self, network, momentum=0.9997):
        super(EmaNetwork, self).__init__()
        self.network = deepcopy(network)
        self.momentum = momentum

    def _update(self, network, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.network.state_dict().values(), network.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, network):
        self._update(network, update_fn=lambda e, m: self.momentum * e + (1. - self.momentum) * m)

    def set(self, network):
        self._update(network, update_fn=lambda e, m: m)

    def forward(self, *args):
        return self.network(*args)


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def generate_G_from_H_batched(H, variable_weight=False):
    """
    calculate batched G from hypergraph incidence matrix H
    :param H: hypergraph incidence matrix H with shape [B, N, E]
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G with shape [B, N, N]
    """
    return _generate_G_from_H_batched(H, variable_weight)


# numpy
# def _generate_G_from_H(H, variable_weight=False):
#     """
#     calculate G from hypgraph incidence matrix H
#     :param H: hypergraph incidence matrix H
#     :param variable_weight: whether the weight of hyperedge is variable
#     :return: G
#     """
#     H = np.array(H)
#     n_edge = H.shape[1]
#     # the weight of the hyperedge
#     W = np.ones(n_edge)
#     # the degree of the node
#     DV = np.sum(H * W, axis=1)
#     # the degree of the hyperedge
#     DE = np.sum(H, axis=0)

#     invDE = np.mat(np.diag(np.power(DE, -1)))
#     DV2 = np.mat(np.diag(np.power(DV, -0.5)))
#     W = np.mat(np.diag(W))
#     H = np.mat(H)
#     HT = H.T

#     if variable_weight:
#         DV2_H = DV2 * H
#         invDE_HT_DV2 = invDE * HT * DV2
#         return DV2_H, W, invDE_HT_DV2
#     else:
#         G = DV2 * H * W * invDE * HT * DV2
#         return G

# tensor
def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = torch.ones(n_edge, device=H.device)
    # the degree of the node
    DV = torch.sum(H, 1)
    # the degree of the hyperedge
    DE = torch.sum(H, 0)

    invDE = torch.diag(torch.pow(DE, -1))
    DV2 = torch.diag(torch.pow(DV, -0.5))
    W = torch.diag(W)
    HT = H.t()

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        # return DV2_H, W, invDE_HT_DV2
        return DV2_H, invDE_HT_DV2
    else:
        G = torch.matmul(DV2, H)
        G = torch.matmul(G, W)
        G = torch.matmul(G, invDE)
        G = torch.matmul(G, HT)
        G = torch.matmul(G, DV2)
        return G


def _generate_G_from_H_batched(H, variable_weight=False):
    """
    calculate batched G from hypergraph incidence matrix H
    :param H: hypergraph incidence matrix H with shape [B, N, E]
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G with shape [B, N, N]
    """
    if variable_weight:
        raise NotImplementedError("Batched variable hyperedge weights are not implemented.")

    eps = 1e-6
    DV = torch.sum(H, dim=2)
    DE = torch.sum(H, dim=1)

    DV_inv_sqrt = torch.clamp(DV, min=eps).pow(-0.5)
    DE_inv = torch.clamp(DE, min=eps).pow(-1.0)

    H_de = H * DE_inv.unsqueeze(1)
    G = torch.bmm(H_de, H.transpose(1, 2))
    G = DV_inv_sqrt.unsqueeze(2) * G * DV_inv_sqrt.unsqueeze(1)
    return G


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x


class HGNN_fc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HGNN_fc, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)


class HGNN_embedding(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.0):
        super(HGNN_embedding, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.hgc2(x, G))
        return x


class HGNN_classifier(nn.Module):
    def __init__(self, n_hid, n_class):
        super(HGNN_classifier, self).__init__()
        self.fc1 = nn.Linear(n_hid, n_class)

    def forward(self, x):
        x = self.fc1(x)
        return x


class HypergraphTransformer(nn.Module):
    def __init__(self, feature_dim, attn_dim):
        super(HypergraphTransformer, self).__init__()
        self.feature_dim = feature_dim
        self.attn_dim = attn_dim
        self.norm = nn.LayerNorm(feature_dim, eps=1e-6)
        self.q_proj = nn.Linear(feature_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(feature_dim, attn_dim, bias=False)
        self.alpha_logit = nn.Parameter(torch.tensor(-0.8473))  # alpha ~= 0.3
        self._monitor_state = {}

    @staticmethod
    def _tensor_stats(tensor, prefix):
        tensor = tensor.detach().float()
        return {
            '{}_mean'.format(prefix): tensor.mean().item(),
            '{}_std'.format(prefix): tensor.std(unbiased=False).item(),
            '{}_abs_mean'.format(prefix): tensor.abs().mean().item(),
            '{}_rms'.format(prefix): tensor.pow(2).mean().sqrt().item(),
            '{}_min'.format(prefix): tensor.min().item(),
            '{}_max'.format(prefix): tensor.max().item(),
        }

    def _grad_stat_hook(self, name):
        def hook(grad):
            self._monitor_state.update(self._tensor_stats(grad, '{}_grad'.format(name)))
        return hook

    def get_monitor_stats(self):
        stats = dict(self._monitor_state)
        stats['alpha_logit'] = self.alpha_logit.detach().item()
        stats['alpha'] = torch.sigmoid(self.alpha_logit.detach()).item()
        stats['alpha_logit_grad'] = 0.0 if self.alpha_logit.grad is None else self.alpha_logit.grad.detach().item()
        stats['q_proj_weight_rms'] = self.q_proj.weight.detach().float().pow(2).mean().sqrt().item()
        stats['k_proj_weight_rms'] = self.k_proj.weight.detach().float().pow(2).mean().sqrt().item()
        if self.q_proj.weight.grad is None:
            stats['q_proj_weight_grad_rms'] = 0.0
        else:
            stats['q_proj_weight_grad_rms'] = self.q_proj.weight.grad.detach().float().pow(2).mean().sqrt().item()
        if self.k_proj.weight.grad is None:
            stats['k_proj_weight_grad_rms'] = 0.0
        else:
            stats['k_proj_weight_grad_rms'] = self.k_proj.weight.grad.detach().float().pow(2).mean().sqrt().item()
        return stats

    def forward(self, V, H_g):
        """
        :param V: stage-4 node features with shape [B, N, D]
        :param H_g: global hypergraph prior with shape [N, E]
        :return: refined incidence H_ref [B, N, E], refined propagation G_ref [B, N, N]
        """
        V_attn = self.norm(V)
        Q = self.q_proj(V_attn)
        K = self.k_proj(V_attn)
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.attn_dim)
        A = F.softmax(attn_scores, dim=-1)

        H_g_batch = H_g.unsqueeze(0).expand(V.size(0), -1, -1)
        H_local = torch.matmul(A, H_g_batch)
        alpha = torch.sigmoid(self.alpha_logit)
        H_ref = (1 - alpha) * H_g_batch + alpha * H_local
        H_delta = H_ref - H_g_batch
        H_g_rms = H_g_batch.detach().float().pow(2).mean().sqrt()
        attn_prob = A.detach().clamp_min(1e-8)

        self._monitor_state = {}
        self._monitor_state.update(self._tensor_stats(attn_scores, 'attn_scores'))
        self._monitor_state.update(self._tensor_stats(Q, 'Q'))
        self._monitor_state.update(self._tensor_stats(K, 'K'))
        self._monitor_state.update(self._tensor_stats(H_g_batch, 'H_g_batch'))
        self._monitor_state.update(self._tensor_stats(H_ref, 'H_ref'))
        self._monitor_state.update(self._tensor_stats(H_delta, 'H_ref_delta'))
        self._monitor_state['alpha'] = alpha.detach().item()
        self._monitor_state['alpha_logit'] = self.alpha_logit.detach().item()
        self._monitor_state['attn_entropy'] = -(attn_prob * attn_prob.log()).sum(dim=-1).mean().item()
        self._monitor_state['H_ref_delta_rel_rms'] = H_delta.detach().float().pow(2).mean().sqrt().div(H_g_rms + 1e-6).item()

        if torch.is_grad_enabled():
            Q.register_hook(self._grad_stat_hook('Q'))
            K.register_hook(self._grad_stat_hook('K'))

        G_ref = generate_G_from_H_batched(H_ref)
        return H_ref, G_ref


class HGNN(nn.Module):
    def __init__(self, in_ch, out_ch, n_hid):
        super(HGNN, self).__init__()
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, out_ch)
        # self.hgc3 = HGNN_conv(n_hid, out_ch)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        # x = F.relu(self.hgc2(x, G))
        x = self.hgc2(x, G)
        return x


class HGNN_Model(nn.Module):
    def __init__(self, input_dim, num_classes=80):
        super(HGNN_Model, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes  # 保存类别数
        # self.stage_2_hgnn = HGNN(512,512,512)
        self.stage_3_hgnn = HGNN(1024, 1024, 1024)
        self.stage_4_hgnn = HGNN(2048, 2048, 2048)
        self.stage_4_refiner = HypergraphTransformer(feature_dim=self.input_dim, attn_dim=512)
        # self.learningMatrix = LearningMatrix()
        # H = np.load('/raid/ocr/lw/SSGRL-HGNN/data/coco/graph/conditional_prob_hyperG_top20_divide_80.npy')
        # self.label_embedding = torch.Tensor(cPickle.load(open('/raid/ocr/lw/ML-GCN/data/coco/coco_glove_word2vec.pkl', 'rb')))
        self.H = nn.Parameter(self.load_features())  # torch.Tensor(generate_G_from_H(H))
        # self.H = torch.eye(80).cuda()

    def load_features(self):
        # 根据num_classes判断数据集并加载对应的词向量文件
        if self.num_classes == 80:  # COCO2014有80个类别
            word_file_path = '/media/ubuntu2/A/coco2014/vectors.npy'
        elif self.num_classes == 200:  # VG有200个类别
            word_file_path = '/home/sx639/GZS/vg/vg_200_vector.npy'
        elif self.num_classes == 20:  # VOC2007有20个类别
            word_file_path = '/home/sx639/GZS/voc2007/voc07_vector.npy'
        else:
            raise ValueError(f"Unsupported number of classes: {self.num_classes}")

        # 加载词向量文件
        features = np.load(word_file_path).astype(np.float32)
        return torch.from_numpy(features)

    def forward(self, stage_3_input, input):
        batch_size = input.size()[0]
        node_num = input.size()[1]
        H_g = torch.sigmoid(self.H)
        global_G = generate_G_from_H(H_g)

        # stage 3
        stage_3_batch_aog_nodes = stage_3_input.view(batch_size, node_num, self.input_dim // 2)
        stage_3_batch_aog_nodes = self.stage_3_hgnn(stage_3_batch_aog_nodes, global_G)

        # stage 4
        stage_4_batch_aog_nodes = input.view(batch_size, node_num, self.input_dim)
        _, G_ref = self.stage_4_refiner(stage_4_batch_aog_nodes, H_g)
        stage_4_batch_aog_nodes = self.stage_4_hgnn(stage_4_batch_aog_nodes, G_ref)

        batch_aog_nodes = torch.cat((stage_3_batch_aog_nodes, stage_4_batch_aog_nodes), 2)

        return batch_aog_nodes

    def get_refiner_monitor_stats(self):
        stats = self.stage_4_refiner.get_monitor_stats()
        H_g = torch.sigmoid(self.H.detach()).float()
        stats['global_H_mean'] = H_g.mean().item()
        stats['global_H_std'] = H_g.std(unbiased=False).item()
        stats['global_H_rms'] = H_g.pow(2).mean().sqrt().item()
        if self.H.grad is None:
            stats['global_H_grad_rms'] = 0.0
        else:
            H_grad = self.H.grad.detach().float()
            stats['global_H_grad_rms'] = H_grad.pow(2).mean().sqrt().item()
        return stats


class AdaHGNN(nn.Module):
    def __init__(self, image_feature_dim, output_dim, word_features, args, num_classes=80, word_feature_dim=300):
        super(AdaHGNN, self).__init__()
        self.resnet_101 = resnet101()

        self.num_classes = args.classNum
        self.word_feature_dim = word_feature_dim
        self.image_feature_dim = image_feature_dim

        self.word_semantic = semantic(num_classes=self.num_classes,
                                      image_feature_dim=self.image_feature_dim,
                                      word_feature_dim=self.word_feature_dim)

        self.word_features = word_features
        self._word_features = self.load_features()

        # 传递num_classes给HGNN_Model
        self.hgnn_model = HGNN_Model(input_dim=self.image_feature_dim, num_classes=self.num_classes)

        self.output_dim = output_dim
        self.fc = nn.Linear(5120, self.output_dim)
        self.classifiers = Classifier_Layer(self.num_classes, self.output_dim)
        # self.test_linear = nn.Linear(2048, self.num_classes)

    def forward(self, x):
        batch_size = x.size()[0]
        feature_3, feature_4 = self.resnet_101(x)
        stage_3_input, stage_4_input = self.word_semantic(batch_size,
                                                          feature_3,
                                                          feature_4,
                                                          self._word_features)
        hgnn_feature = self.hgnn_model(stage_3_input, stage_4_input)
        output = torch.cat(
            (hgnn_feature.view(batch_size * self.num_classes, -1), stage_4_input.view(-1, self.image_feature_dim)), 1)
        output = self.fc(output)
        output = torch.tanh(output)
        output = output.contiguous().view(batch_size, self.num_classes, self.output_dim)
        result = self.classifiers(output)
        return result

    def get_refiner_monitor_stats(self):
        return self.hgnn_model.get_refiner_monitor_stats()

    def load_features(self):
        return Parameter(torch.from_numpy(self.word_features.astype(np.float32)))
