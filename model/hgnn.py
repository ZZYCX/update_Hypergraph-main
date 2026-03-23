import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from .backbone.resnet import resnet101
from .semantic import semantic
from .classifier_layer import Classifier_Layer

import math
from torch.nn.parameter import Parameter
from six.moves import cPickle
import time
import torchvision.models as models


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
    # W = torch.ones(n_edge)
    # the degree of the node
    DV = torch.sum(H, 1)
    # the degree of the hyperedge
    DE = torch.sum(H, 0)

    invDE = torch.diag(torch.pow(DE, -1))
    DV2 = torch.diag(torch.pow(DV, -0.5))
    # W = torch.diag(W)
    HT = H.t()

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        # return DV2_H, W, invDE_HT_DV2
        return DV2_H, invDE_HT_DV2
    else:
        G = torch.matmul(DV2, H)
        # G = torch.matmul(G, W)
        G = torch.matmul(G, invDE)
        G = torch.matmul(G, HT)
        G = torch.matmul(G, DV2)
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
    def __init__(self, input_dim):
        super(HGNN_Model, self).__init__()
        self.input_dim = input_dim

        # self.stage_2_hgnn = HGNN(512,512,512)
        self.stage_3_hgnn = HGNN(1024, 1024, 1024)
        self.stage_4_hgnn = HGNN(2048, 2048, 2048)
        # self.learningMatrix = LearningMatrix()
        # H = np.load('/raid/ocr/lw/SSGRL-HGNN/data/coco/graph/conditional_prob_hyperG_top20_divide_80.npy')
        # self.label_embedding = torch.Tensor(cPickle.load(open('/raid/ocr/lw/ML-GCN/data/coco/coco_glove_word2vec.pkl', 'rb')))
        self.H = nn.Parameter(self.load_features())  # torch.Tensor(generate_G_from_H(H))
        # self.H = torch.eye(80).cuda()

    def load_features(self):
        return torch.from_numpy(np.load('/hy-tmp/coco2014/vectors.npy').astype(np.float32))

    def forward(self, stage_3_input, input):
        batch_size = input.size()[0]
        node_num = input.size()[1]
        device = input.get_device()

        # stage 3
        stage_3_batch_aog_nodes = stage_3_input.view(batch_size, node_num, self.input_dim // 2)
        stage_3_batch_aog_nodes = self.stage_3_hgnn(stage_3_batch_aog_nodes, generate_G_from_H(torch.sigmoid(self.H)))

        # stage 4
        stage_4_batch_aog_nodes = input.view(batch_size, node_num, self.input_dim)
        stage_4_batch_aog_nodes = self.stage_4_hgnn(stage_4_batch_aog_nodes, generate_G_from_H(torch.sigmoid(self.H)))

        batch_aog_nodes = torch.cat((stage_3_batch_aog_nodes, stage_4_batch_aog_nodes), 2)

        return batch_aog_nodes


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

        self.hgnn_model = HGNN_Model(input_dim=self.image_feature_dim)

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
                                                          torch.tensor(self._word_features))
        hgnn_feature = self.hgnn_model(stage_3_input, stage_4_input)
        output = torch.cat(
            (hgnn_feature.view(batch_size * self.num_classes, -1), stage_4_input.view(-1, self.image_feature_dim)), 1)
        output = self.fc(output)
        output = torch.tanh(output)
        output = output.contiguous().view(batch_size, self.num_classes, self.output_dim)
        result = self.classifiers(output)
        return result

    def load_features(self):
        return Parameter(torch.from_numpy(self.word_features.astype(np.float32)))