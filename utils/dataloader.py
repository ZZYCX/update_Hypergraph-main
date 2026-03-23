import os
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets.coco2014 import COCO2014
from datasets.vg import VG
from datasets.voc2007 import VOC2007
from config import prefixPathCOCO, prefixPathVOC2007, prefixPathVG

def get_graph_and_word_file(args, labels):
    def get_graph_file(labels):

        graph = np.zeros((labels.shape[1], labels.shape[1]), dtype=np.float64)

        for index in range(labels.shape[0]):
            indexs = np.where(labels[index] == 1)[0]
            for i in indexs:
                for j in indexs:
                    graph[i, j] += 1

        for i in range(labels.shape[1]):
            graph[i] /= graph[i, i]

        np.nan_to_num(graph)

        return graph

    if args.dataset == 'COCO2014':
        WordFilePath = '/media/ubuntu2/A/coco2014/vectors.npy'
    elif args.dataset == 'VG':
        WordFilePath = '/home/sx639/GZS/vg/vg_200_vector.npy'
    elif args.dataset == 'VOC2007':
        WordFilePath = '/home/sx639/GZS/voc2007/voc07_vector.npy'

    GraphFile = get_graph_file(labels)
    WordFile = np.load(WordFilePath)

    return GraphFile, WordFile



def get_data_path(dataset):
    if dataset == 'COCO2014':
        prefixPath = prefixPathCOCO
        train_dir, train_anno, train_label = os.path.join(prefixPath, 'train2014'), os.path.join(prefixPath,
                                                                                                 'annotations/instances_train2014.json'), 'train_label_vectors.npy'
        test_dir, test_anno, test_label = os.path.join(prefixPath, 'val2014'), os.path.join(prefixPath,
                                                                                            'annotations/instances_val2014.json'), 'val_label_vectors.npy'

    elif dataset == 'VOC2007':
        prefixPath = prefixPathVOC2007
        train_dir, train_anno, train_label = os.path.join(prefixPath, 'JPEGImages'), os.path.join(prefixPath,
                                                                                                  'ImageSets/Main/trainval.txt'), os.path.join(
            prefixPath, 'Annotations')
        test_dir, test_anno, test_label = os.path.join(prefixPath, 'JPEGImages'), os.path.join(prefixPath,
                                                                                               'ImageSets/Main/test.txt'), os.path.join(
            prefixPath, 'Annotations')
    elif dataset == 'VG':
        prefixPath = prefixPathVG
        train_dir, train_anno, train_label = os.path.join(prefixPath,
                                                          'VG_100K'), os.path.join(prefixPath,
                                                                                   'train_list_500.txt'), os.path.join(
            prefixPath, 'vg_category_200_labels_index.json')
        test_dir, test_anno, test_label = os.path.join(prefixPath,
                                                       'VG_100K'), os.path.join(prefixPath,
                                                                                'test_list_500.txt'), os.path.join(
            prefixPath, 'vg_category_200_labels_index.json')

    return train_dir, train_anno, train_label, test_dir, test_anno, test_label




def get_data_loader(args):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    scale_size = args.scale_size
    crop_size = args.crop_size

    train_data_transform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                               transforms.RandomChoice([
                                                   transforms.RandomCrop(640),
                                                   transforms.RandomCrop(576),
                                                   transforms.RandomCrop(512),
                                                   transforms.RandomCrop(384),
                                                   transforms.RandomCrop(320)]),
                                               transforms.Resize((crop_size, crop_size)),
                                               transforms.ToTensor(),
                                               normalize])

    test_data_transform = transforms.Compose([transforms.Resize((crop_size, crop_size)),
                                              transforms.ToTensor(),
                                              normalize])

    train_dir, train_anno, train_label, \
        test_dir, test_anno, test_label = get_data_path(args.dataset)

    if args.dataset == 'COCO2014':
        print("==> Loading COCO2014...")
        train_set = COCO2014('train', train_dir, train_anno, train_label, input_transform=train_data_transform,
                             label_proportion=args.label_proportion)
        test_set = COCO2014('val', test_dir, test_anno, test_label, input_transform=test_data_transform)

    elif args.dataset == 'VG':
        print("==> Loading VG...")
        train_set = VG('train',
                       train_dir, train_anno, train_label,
                       input_transform=train_data_transform, label_proportion=args.label_proportion)
        test_set = VG('val',
                      test_dir, test_anno, test_label,
                      input_transform=test_data_transform)

    elif args.dataset == 'VOC2007':
        print("==> Loading VOC2007...")
        train_set = VOC2007('train',
                            train_dir, train_anno, train_label,
                            input_transform=train_data_transform, label_proportion=args.label_proportion)
        test_set = VOC2007('val',
                           test_dir, test_anno, test_label,
                           input_transform=test_data_transform)

    else:
        print('%s Dataset Not Found' % args.dataset)
        exit(1)
    # train_ddp_sampler = DistributedSaplmpler(train_set, num_replicas=args.world_size, rank=rank, shuffle=False, drop_last=False)
    # test_ddp_sampler = DistributedSamer(test_set, num_replicas=args.world_size, rank=rank, shuffle=False, drop_last=False)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              pin_memory=True,
                              drop_last=True,
                              # sampler=train_ddp_sampler,
                              )
    test_loader = DataLoader(dataset=test_set,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.workers,
                             pin_memory=True,
                             drop_last=True,
                             # sampler=test_ddp_sampler
                             )

    return train_loader, test_loader

