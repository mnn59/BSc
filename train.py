import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling import TransCASCADE
from trainer import trainer_polyp, trainer_cascade

parser = argparse.ArgumentParser()
parser.add_argument('--img_root', type=str, default='/content/drive/MyDrive/MyBScProject/project_TransUNet/data/Polyp/Original/', help='root dir for data')
parser.add_argument('--gt_root', type=str, default='/content/drive/MyDrive/MyBScProject/project_TransUNet/data/Polyp/Ground Truth/', help='root dir for mask')
parser.add_argument('--dataset', type=str, default='Polyp', help='experiment_name')
parser.add_argument('--list_dir', type=str, default='/content/drive/MyDrive/MyTransunet/TransUNet-repo/TransUNet/lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--arch', type=str, default='Transunet', help='architecture name')
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '/content/drive/MyDrive/MyTransunet/TransUNet-repo/data/Synapse/train_npz',
            'list_dir': '/content/drive/MyDrive/MyTransunet/TransUNet-repo/TransUNet/lists/lists_Synapse',
            'num_classes': 9,
        },
        'Polyp': {  # polyp means cvc-clinicdb
          'img_root': '/content/drive/MyDrive/MyBScProject/project_TransUNet/data/Polyp/Original/',
          'gt_root': '/content/drive/MyDrive/MyBScProject/project_TransUNet/data/Polyp/Ground Truth/',
          'num_classes': 2,
        },
        'Kvasir': {
          'img_root': '/content/drive/MyDrive/datasets/Kvasir-SEG/images/',
          'gt_root': '/content/drive/MyDrive/datasets/Kvasir-SEG/masks/',
          'num_classes': 2,
        }, 
        'Ph2': {
          'img_root': '/content/drive/MyDrive/datasets/ph2/trainx/',
          'gt_root': '/content/drive/MyDrive/datasets/ph2/trainy/',
          'num_classes': 2,
        },
        'CVCKvasir':{  # combinitation of Polyp(CVC-ClinicDB) and Kvasir
          'img_root': '/content/drive/MyDrive/datasets/CVC-Kvasir/images/',
          'gt_root': '/content/drive/MyDrive/datasets/CVC-Kvasir/masks/',
          'num_classes': 2,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.img_root = dataset_config[dataset_name]['img_root']
    args.gt_root = dataset_config[dataset_name]['gt_root']
    args.is_pretrain = True
    args.exp = ('TU_' if args.arch == 'Transunet' else 'TransCASCADE_') + dataset_name + str(args.img_size)  # Both Transunet and TransCASCADE
    snapshot_path = "/content/drive/MyDrive/MyTransunet/TransUNet-repo/model/{}/{}".format(args.exp, ('TU' if args.arch == 'Transunet' else 'TransCASCADE'))  # Both Transunet and TransCASCADE
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    nets = {
        'Transunet': ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda(),
        'Transcascade': TransCASCADE(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    }
    net = nets[args.arch]  # Both Transunet and TransCASCADE
    net.load_from(weights=np.load(config_vit.pretrained_path))

    trainer_ = {
        'Transunet': trainer_polyp,
        'Transcascade': trainer_cascade
    }
    trainer = {'Polyp': trainer_[args.arch], 'Kvasir': trainer_[args.arch], 'Ph2': trainer_[args.arch], 'CVCKvasir': trainer_[args.arch]}  # Both Transunet and TransCASCADE
    trainer[dataset_name](args, net, snapshot_path)