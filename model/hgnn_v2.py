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
    def __init__(self, feature_dim, edge_dim, attn_dim=None, eps=1e-6):
        super(HypergraphTransformer, self).__init__()
        self.feature_dim = feature_dim
        self.edge_dim = edge_dim
        self.attn_dim = attn_dim if attn_dim is not None else feature_dim
        self.monitor_eps = eps
        self.norm = nn.LayerNorm(feature_dim, eps=eps)
        self.q_proj = nn.Linear(feature_dim, self.attn_dim, bias=False)
        self.k_proj = nn.Linear(feature_dim, self.attn_dim, bias=False)
        self.edge_proj = nn.Linear(feature_dim, edge_dim, bias=True)
        self.beta_logit = nn.Parameter(torch.tensor(-1.4))
        self._last_monitor_stats = {}

    def forward(self, V, H_g):
        """
        :param V: stage-4 node features with shape [B, N, D]
        :param H_g: global hypergraph prior with shape [N, E]
        :return: refined incidence H_ref [B, N, E], refined propagation G_ref [B, N, N]
        """
        if V.dim() != 3:
            raise ValueError("Expected V to have shape [B, N, D].")
        if H_g.dim() != 2:
            raise ValueError("Expected H_g to have shape [N, E].")
        if V.size(-1) != self.feature_dim:
            raise ValueError(
                f"Expected V feature dim {self.feature_dim}, got {V.size(-1)}."
            )
        if V.size(1) != H_g.size(0):
            raise ValueError(
                f"Node count mismatch between V ({V.size(1)}) and H_g ({H_g.size(0)})."
            )
        if H_g.size(1) != self.edge_dim:
            raise ValueError(
                f"Expected H_g edge dim {self.edge_dim}, got {H_g.size(1)}."
            )

        V_attn = self.norm(V)
        Q = self.q_proj(V_attn)
        K = self.k_proj(V_attn)
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.attn_dim)
        attn = F.softmax(attn_scores, dim=-1)

        context = torch.matmul(attn, V_attn)
        delta_h = torch.tanh(self.edge_proj(context))
        beta = torch.sigmoid(self.beta_logit)
        H_g_batch = H_g.unsqueeze(0).expand(V.size(0), -1, -1)
        H_raw = H_g_batch + beta * delta_h
        H_ref = torch.clamp(H_raw, min=0.0, max=1.0)
        G_ref = generate_G_from_H_batched(H_ref)

        with torch.no_grad():
            clamp_upper_ratio = (H_raw > 1.0).float().mean()
            clamp_lower_ratio = (H_raw < 0.0).float().mean()
            clamp_ratio = ((H_raw > 1.0) | (H_raw < 0.0)).float().mean()
            h_ref_shift_ratio = (
                torch.abs(H_ref - H_g_batch)
                / torch.clamp(torch.abs(H_g_batch), min=self.monitor_eps)
            ).mean()
            sample_var = H_ref.var(dim=0, unbiased=False).mean()
            self._last_monitor_stats = {
                'beta': beta.detach().item(),
                'delta_h_mean_abs': delta_h.detach().abs().mean().item(),
                'clamp_upper_ratio': clamp_upper_ratio.detach().item(),
                'clamp_lower_ratio': clamp_lower_ratio.detach().item(),
                'clamp_ratio': clamp_ratio.detach().item(),
                'h_ref_shift_ratio': h_ref_shift_ratio.detach().item(),
                'sample_var': sample_var.detach().item(),
            }

        return H_ref, G_ref

    def get_monitor_stats(self):
        return dict(self._last_monitor_stats)


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
        # self.learningMatrix = LearningMatrix()
        # H = np.load('/raid/ocr/lw/SSGRL-HGNN/data/coco/graph/conditional_prob_hyperG_top20_divide_80.npy')
        # self.label_embedding = torch.Tensor(cPickle.load(open('/raid/ocr/lw/ML-GCN/data/coco/coco_glove_word2vec.pkl', 'rb')))
        H_init = self.load_features()
        self.H = nn.Parameter(H_init)  # torch.Tensor(generate_G_from_H(H))
        self.stage_4_refiner = HypergraphTransformer(
            feature_dim=self.input_dim,
            edge_dim=H_init.size(1),
            attn_dim=512,
        )
        self._last_refinement_monitor_stats = {}
        # self.H = torch.eye(80).cuda()

    def load_features(self):
        # 根据num_classes判断数据集并加载对应的词向量文件
        if self.num_classes == 80:  # COCO2014有80个类别
            word_file_path = '/home/sx639/GZS/coco2014/vectors.npy'
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
        self._last_refinement_monitor_stats = self.stage_4_refiner.get_monitor_stats()
        stage_4_batch_aog_nodes = self.stage_4_hgnn(stage_4_batch_aog_nodes, G_ref)

        batch_aog_nodes = torch.cat((stage_3_batch_aog_nodes, stage_4_batch_aog_nodes), 2)

        return batch_aog_nodes

    def get_refinement_monitor_stats(self):
        return dict(self._last_refinement_monitor_stats)


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

    def get_refinement_monitor_stats(self):
        return self.hgnn_model.get_refinement_monitor_stats()

    def load_features(self):
        return Parameter(torch.from_numpy(self.word_features.astype(np.float32)))
