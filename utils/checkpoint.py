import os
import gc
import shutil

import torch


def load_pretrained_model(model, args):
    modelDict = model.resnet_101.state_dict()

    pretrainedModel = torch.load(args.pretrain_model)
    pretrainedDict = {}
    for k, v in pretrainedModel.items():
        # print(k)
        if k.startswith('fc'):
            continue
        pretrainedDict[k] = v
    modelDict.update(pretrainedDict)
    model.resnet_101.load_state_dict(modelDict)

    del pretrainedModel
    del pretrainedDict
    gc.collect()

    return model


def save_checkpoint(args, state, isBest):

    outputPath = os.path.join('exp/checkpoint/', args.post)
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    torch.save(state, os.path.join(outputPath, 'Checkpoint_Current.pth'))
    if isBest:
        torch.save(state, os.path.join(outputPath, 'Checkpoint_Best.pth'))