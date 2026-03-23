import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torch


# v1
class semantic(nn.Module):
    def __init__(self, num_classes, image_feature_dim, word_feature_dim, intermediary_dim=1024):
        super(semantic, self).__init__()

        # bilinear pooling
        self.num_classes = num_classes
        self.image_feature_dim = image_feature_dim
        self.word_feature_dim = word_feature_dim
        self.intermediary_dim = intermediary_dim

        # stage_1
        # self.stage_1_fc_1 = nn.Linear(self.image_feature_dim // 8, self.intermediary_dim // 8, bias=False)
        # self.stage_1_fc_2 = nn.Linear(self.word_feature_dim, self.intermediary_dim // 8, bias=False)
        # self.stage_1_fc_3 = nn.Linear(self.intermediary_dim // 8, self.intermediary_dim // 8)
        # self.stage_1_fc_a = nn.Linear(self.intermediary_dim // 8, 1)

        # # stage_2
        # self.stage_2_fc_1 = nn.Linear(self.image_feature_dim // 4, self.intermediary_dim // 4, bias=False)
        # self.stage_2_fc_2 = nn.Linear(self.word_feature_dim, self.intermediary_dim // 4, bias=False)
        # self.stage_2_fc_3 = nn.Linear(self.intermediary_dim // 4, self.intermediary_dim // 4)
        # self.stage_2_fc_a = nn.Linear(self.intermediary_dim // 4, 1)

        # stage_3
        self.stage_3_fc_1 = nn.Linear(self.image_feature_dim // 2, self.intermediary_dim // 2, bias=False)
        self.stage_3_fc_2 = nn.Linear(self.word_feature_dim, self.intermediary_dim // 2, bias=False)
        self.stage_3_fc_3 = nn.Linear(self.intermediary_dim // 2, self.intermediary_dim // 2)
        self.stage_3_fc_a = nn.Linear(self.intermediary_dim // 2, 1)

        # stage_4
        self.fc_1 = nn.Linear(self.image_feature_dim, self.intermediary_dim, bias=False)
        self.fc_2 = nn.Linear(self.word_feature_dim, self.intermediary_dim, bias=False)
        self.fc_3 = nn.Linear(self.intermediary_dim, self.intermediary_dim)
        self.fc_a = nn.Linear(self.intermediary_dim, 1)

    def forward(self, batch_size, stage_3_img_feature_map, stage_4_img_feature_map, word_features):
        # stage 3
        img_feature_map = stage_3_img_feature_map
        convsize = img_feature_map.size()[3]

        img_feature_map = torch.transpose(torch.transpose(img_feature_map, 1, 2), 2, 3)
        f_wh_feature = img_feature_map.contiguous().view(batch_size * convsize * convsize, -1)
        f_wh_feature = self.stage_3_fc_1(f_wh_feature).view(batch_size * convsize * convsize, 1, -1).repeat(1,
                                                                                                            self.num_classes,
                                                                                                            1)

        f_wd_feature = self.stage_3_fc_2(word_features).view(1, self.num_classes, self.intermediary_dim // 2).repeat(
            batch_size * convsize * convsize, 1, 1)
        lb_feature = self.stage_3_fc_3(torch.tanh(f_wh_feature * f_wd_feature).view(-1, self.intermediary_dim // 2))
        coefficient = self.stage_3_fc_a(lb_feature)
        coefficient = torch.transpose(
            torch.transpose(coefficient.view(batch_size, convsize, convsize, self.num_classes), 2, 3), 1, 2).view(
            batch_size, self.num_classes, -1)

        coefficient = F.softmax(coefficient, dim=2)
        coefficient = coefficient.view(batch_size, self.num_classes, convsize, convsize)
        coefficient = torch.transpose(torch.transpose(coefficient, 1, 2), 2, 3)
        coefficient = coefficient.view(batch_size, convsize, convsize, self.num_classes, 1).repeat(1, 1, 1, 1,
                                                                                                   self.image_feature_dim // 2)
        img_feature_map = img_feature_map.view(batch_size, convsize, convsize, 1, self.image_feature_dim // 2).repeat(1,
                                                                                                                      1,
                                                                                                                      1,
                                                                                                                      self.num_classes,
                                                                                                                      1) * coefficient
        stage_3_img_feature_map = torch.sum(torch.sum(img_feature_map, 1), 1)

        # stage 4
        img_feature_map = stage_4_img_feature_map
        convsize = img_feature_map.size()[3]

        img_feature_map = torch.transpose(torch.transpose(img_feature_map, 1, 2), 2, 3)
        f_wh_feature = img_feature_map.contiguous().view(batch_size * convsize * convsize, -1)
        f_wh_feature = self.fc_1(f_wh_feature).view(batch_size * convsize * convsize, 1, -1).repeat(1, self.num_classes,
                                                                                                    1)

        f_wd_feature = self.fc_2(word_features).view(1, self.num_classes, self.intermediary_dim).repeat(
            batch_size * convsize * convsize, 1, 1)
        lb_feature = self.fc_3(torch.tanh(f_wh_feature * f_wd_feature).view(-1, self.intermediary_dim))
        coefficient = self.fc_a(lb_feature)
        coefficient = torch.transpose(
            torch.transpose(coefficient.view(batch_size, convsize, convsize, self.num_classes), 2, 3), 1, 2).view(
            batch_size, self.num_classes, -1)

        coefficient = F.softmax(coefficient, dim=2)
        coefficient = coefficient.view(batch_size, self.num_classes, convsize, convsize)
        coefficient = torch.transpose(torch.transpose(coefficient, 1, 2), 2, 3)
        coefficient = coefficient.view(batch_size, convsize, convsize, self.num_classes, 1).repeat(1, 1, 1, 1,
                                                                                                   self.image_feature_dim)
        img_feature_map = img_feature_map.view(batch_size, convsize, convsize, 1, self.image_feature_dim).repeat(1, 1,
                                                                                                                 1,
                                                                                                                 self.num_classes,
                                                                                                                 1) * coefficient
        stage_4_img_feature_map = torch.sum(torch.sum(img_feature_map, 1), 1)

        return stage_3_img_feature_map, stage_4_img_feature_map

# v2
# class semantic(nn.Module):
#     def __init__(self, num_classes, image_feature_dim, word_feature_dim, intermediary_dim=1024):
#         super(semantic, self).__init__()

#         # bilinear pooling
#         self.num_classes = num_classes
#         self.image_feature_dim = image_feature_dim
#         self.word_feature_dim = word_feature_dim
#         self.intermediary_dim = intermediary_dim

#         # stage_1
#         # self.stage_1_fc_1 = nn.Linear(self.image_feature_dim // 8, self.intermediary_dim // 8, bias=False)
#         # self.stage_1_fc_2 = nn.Linear(self.word_feature_dim, self.intermediary_dim // 8, bias=False)
#         # self.stage_1_fc_3 = nn.Linear(self.intermediary_dim // 8, self.intermediary_dim // 8)
#         # self.stage_1_fc_a = nn.Linear(self.intermediary_dim // 8, 1)

#         # # stage_2
#         # self.stage_2_fc_1 = nn.Linear(self.image_feature_dim // 4, self.intermediary_dim // 4, bias=False)
#         # self.stage_2_fc_2 = nn.Linear(self.word_feature_dim, self.intermediary_dim // 4, bias=False)
#         # self.stage_2_fc_3 = nn.Linear(self.intermediary_dim // 4, self.intermediary_dim // 4)
#         # self.stage_2_fc_a = nn.Linear(self.intermediary_dim // 4, 1)

#         # stage_3
#         self.stage_3_fc = nn.Sequential(
#             nn.Linear(300, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1024)
#         )

#         # stage_4
#         self.stage_4_fc = nn.Sequential(
#             nn.Linear(300, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 2048)
#         )

#     def forward(self,batch_size,stage_3_img_feature_map,stage_4_img_feature_map, word_features):
#         # stage 3
#         stage_3_img_feature_map = stage_3_img_feature_map.view(batch_size, 1, 1024)
#         stage_3_img_feature_map = stage_3_img_feature_map * self.stage_3_fc(word_features)

#         # stage 4
#         stage_4_img_feature_map = stage_4_img_feature_map.view(batch_size, 1, 2048)
#         stage_4_img_feature_map = stage_4_img_feature_map * self.stage_4_fc(word_features)

#         return stage_3_img_feature_map, stage_4_img_feature_map

# v3
# class semantic(nn.Module):
#     def __init__(self, num_classes, image_feature_dim, word_feature_dim, intermediary_dim=1024):
#         super(semantic, self).__init__()

#         # bilinear pooling
#         self.num_classes = num_classes
#         self.image_feature_dim = image_feature_dim
#         self.word_feature_dim = word_feature_dim
#         self.intermediary_dim = intermediary_dim

#         # stage_1
#         # self.stage_1_fc_1 = nn.Linear(self.image_feature_dim // 8, self.intermediary_dim // 8, bias=False)
#         # self.stage_1_fc_2 = nn.Linear(self.word_feature_dim, self.intermediary_dim // 8, bias=False)
#         # self.stage_1_fc_3 = nn.Linear(self.intermediary_dim // 8, self.intermediary_dim // 8)
#         # self.stage_1_fc_a = nn.Linear(self.intermediary_dim // 8, 1)

#         # # stage_2
#         # self.stage_2_fc_1 = nn.Linear(self.image_feature_dim // 4, self.intermediary_dim // 4, bias=False)
#         # self.stage_2_fc_2 = nn.Linear(self.word_feature_dim, self.intermediary_dim // 4, bias=False)
#         # self.stage_2_fc_3 = nn.Linear(self.intermediary_dim // 4, self.intermediary_dim // 4)
#         # self.stage_2_fc_a = nn.Linear(self.intermediary_dim // 4, 1)

#         # stage_3
#         self.stage_3_conv = nn.Sequential(
#             nn.Conv2d(1024, 512, 1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace = True),
#             nn.Conv2d(512, 512, 1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace = True),
#             nn.Conv2d(512, 80, 1),
#             nn.BatchNorm2d(80),
#             nn.ReLU(inplace = True),
#         )

#         # stage_4
#         self.stage_4_conv = nn.Sequential(
#             nn.Conv2d(2048, 1024, 1),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(inplace = True),
#             nn.Conv2d(1024, 1024, 1),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(inplace = True),
#             nn.Conv2d(1024, 80, 1),
#             nn.BatchNorm2d(80),
#             nn.ReLU(inplace = True),
#         )

#         self.softmax = nn.Softmax(dim=2)


#     def forward(self,batch_size,stage_3_img_feature_map,stage_4_img_feature_map, word_features):
#         # stage 3
#         stage_3_class_attention_map = self.stage_3_conv(stage_3_img_feature_map)
#         stage_3_class_attention_map = self.softmax(stage_3_class_attention_map.view(batch_size, 80, 1, -1))
#         stage_3_img_feature_map = stage_3_img_feature_map.view(batch_size, 1, 1024, -1) * stage_3_class_attention_map
#         stage_3_img_feature_map = torch.sum(stage_3_img_feature_map, 3)

#         # stage 4
#         stage_4_class_attention_map = self.stage_4_conv(stage_4_img_feature_map)
#         stage_4_class_attention_map = self.softmax(stage_4_class_attention_map.view(batch_size, 80, 1, -1))
#         stage_4_img_feature_map = stage_4_img_feature_map.view(batch_size, 1, 2048, -1) * stage_4_class_attention_map
#         stage_4_img_feature_map = torch.sum(stage_4_img_feature_map, 3)

#         return stage_3_img_feature_map, stage_4_img_feature_map