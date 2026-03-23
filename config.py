import logging
import warnings
import argparse

warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Dataset Path
# =============================================================================
prefixPathCOCO = '/media/ubuntu2/A/coco2014'
# '/home/gpu4/fsw/data/COCO2014/'
prefixPathVG = '/home/sx639/GZS/vg/'
prefixPathVOC2007 = '/home/sx639/GZS/voc2007/'
# =============================================================================
''
# ClassNum of Dataset
# =============================================================================
_ClassNum = {'COCO2014': 80,
             'VOC2007': 20,
             'VG': 200,
            }
# =============================================================================
def str2bool(input):
    if isinstance(input, bool):
        return input

    if input.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def show_args(args):

    logger.info("==========================================")
    logger.info("==========       CONFIG      =============")
    logger.info("==========================================")

    for arg, content in args.__dict__.items():
        logger.info("{}: {}".format(arg, content))

    logger.info("==========================================")
    logger.info("===========        END        ============")
    logger.info("==========================================")

    logger.info("\n")


def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch multi label Training')
    parser.add_argument('--dataset', type=str, default='COCO2014', choices=['COCO2014', 'VG', 'VOC2007'],
                        help='dataset for training and testing')
    # parser.add_argument('--train_data', default=r'/hy-tmp/coco2014/train2014', metavar='DIR',
    #                     help='path to train dataset')
    # parser.add_argument('--test_data', default=r'/hy-tmp/coco2014/val2014', metavar='DIR',
    #                     help='path to test dataset')
    # parser.add_argument('--trainlist', default=r'/hy-tmp/coco2014/annotations/instances_train2014.json',
    #                     metavar='DIR',
    #                     help='path to train list')
    # parser.add_argument('--testlist', default=r'/hy-tmp/coco2014/annotations/instances_val2014.json', metavar='DIR',
    #                     help='path to test list')
    parser.add_argument('-pm', '--pretrain_model', type=str,
                        default=r'/media/ubuntu2/A/coco2014/resnet101.pth', metavar='PATH',
                        help='path to latest pretrained_model (default: none)')
    # parser.add_argument('-train_label', default=r'/hy-tmp/coco2014/train_label_vectors.npy', type=str,
    #                     metavar='PATH',
    #                     help='path to train label (default: none)')
    # parser.add_argument('-graph_file', default=r'/hy-tmp/coco2014/prob_train.npy', type=str, metavar='PATH',
    #                     help='path to graph (default: none)')
    # parser.add_argument('-word_file', default=r'/hy-tmp/coco2014/vectors.npy', type=str, metavar='PATH',
    #                     help='path to word feature')
    # parser.add_argument('-test_label', default=r'/hy-tmp/coco2014/val_label_vectors.npy', type=str, metavar='PATH',
    #                     help='path to test label (default: none)')

    parser.add_argument('--print_freq', '-p', default=1000, type=int, metavar='N',
                        help='number of print_freq (default: 100)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--crop_size', dest='crop_size', default=448, type=int,
                        help='crop size')
    parser.add_argument('--scale_size', dest='scale_size', default=640, type=int,
                        help='the size of the rescale image')
    parser.add_argument('--latdim', default=2048, type=int,
                        metavar='N', help='dimension of HyperGraph Transformer embeddings (default: 3072)')

    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--step_epoch', default=20, type=int, metavar='N',
                        help='decend the lr in epoch number')

    parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('-lr-scheduler', default='None', type=str, metavar='PATH',
                        help='type of scheduler')
    parser.add_argument('--step-lr-gamma', default=0.1, type=float, metavar='N',
                        help='rate of decend the lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--ema-start', default=0, type=int,
                        metavar='N', help='start of ema (default: 0)')
    parser.add_argument('--ema-momentum', default=0.9997, type=float,
                        metavar='N', help='momentum of ema (default: 0.9997)')

    parser.add_argument('--amp', default='False', type=str2bool,
                        metavar='N', help='use the amp (default: False)')
    parser.add_argument('--frozen_batch_norm', default='False', type=str2bool,
                        metavar='N', help='froze the resnet batch norm (default: False)')
    parser.add_argument('--asl', default='False', type=str2bool,
                        metavar='N', help='use ASL Loss (default: False)')
    parser.add_argument('--gamma-neg', default=2.0, type=float,
                        metavar='N', help='weight of negatives loss (default: 2.0)')
    parser.add_argument('--gamma-pos', default=0.0, type=float,
                        metavar='N', help='weight of positives loss (default: 0.0)')
    parser.add_argument('--loss-clip', default=0.0, type=float,
                        metavar='N', help='clip factor of negatives probability(default: 0.0)')
    parser.add_argument('--dtgfl', default='True', type=str2bool,
                        metavar='N', help='disable torch grad focal loss (default: True)')
    parser.add_argument('--eps', default=0.00001, type=float,
                        metavar='N', help='eps (default: 0.00001)')

    parser.add_argument('--att_head', default=4, type=int,
                        metavar='N', help='number of attention head (default: 2)')
    parser.add_argument('--encoder_layer', default=1, type=int,
                        metavar='N', help='number of encoder layer (default: 1)')
    parser.add_argument('--forward_factor', default=4, type=int,
                        metavar='N', help='hidden dimension scale factor (default: 4)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoint', metavar='PATH',
                        help='path to latest pretrained_model (default: none)')
    parser.add_argument('--resume', default='None', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--evaluate', dest='evaluate', type=str2bool, default='False',
                        help='evaluate model on validation set')
    parser.add_argument('--post', dest='post', type=str, default='AdaHGNN_ViT_3_10',
                        help='postname of save model')

    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-world_size', '--world_size', default=4, type=int, metavar='number of processes')
    parser.add_argument('-nr', '--nr', default=0, type=int, metavar='ranking within the nodes')
    parser.add_argument('--label_proportion', default=0.5, type=float,
                        help='proportion of labels to use for partial label learning (default: 1.0)')

    args = parser.parse_args()
    args.classNum = _ClassNum[args.dataset]

    return args
