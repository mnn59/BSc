import argparse
import logging
import os
import random
import sys
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import SimpleITK as sitk
from datasets.dataset_polyp import PolypDataset
from utils import test_single_volume, calculate_metric_percase
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import TransCASCADE
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from datasets.dataset_polyp import test_dataset, get_loader
from torchvision import transforms
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('--img_root', type=str, default='/content/drive/MyDrive/MyBScProject/project_TransUNet/data/Polyp/Original/', help='root dir for data')
parser.add_argument('--gt_root', type=str, default='/content/drive/MyDrive/MyBScProject/project_TransUNet/data/Polyp/Ground Truth/', help='root dir for mask')
parser.add_argument('--dataset', type=str, default='Polyp', help='experiment_name')
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_save', action="store_true", help='whether to save results during inference')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--arch', type=str, default='Transunet', help='architecture name')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    testloader = get_loader(args.img_root, args.gt_root, split="test", batchsize=1, trainsize=args.img_size, augmentation=False)

    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metriclist = 0.0
    for i_batch, sampled_batch in enumerate(tqdm(testloader)):
        image_batch, label_batch = sampled_batch[0], sampled_batch[1]
        model.eval()

        with torch.no_grad():
            image_batch = image_batch.float().cuda()
            if args.arch == 'Transunet':
                prediction = torch.argmax(torch.softmax(model(image_batch), dim=1), dim=1)
            elif args.arch == 'Transcascade':
                p1, p2, p3, p4 = model(image_batch)
                outputs = p1 + p2 + p3 + p4
                prediction = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        metric_list = []
        for i in range(1, args.num_classes):
            metric_list.append(calculate_metric_percase(prediction.detach().cpu().numpy() == i, label_batch.detach().cpu().numpy() == i))
        
        if test_save_path is not None:
            img = image_batch.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            prd = prediction.permute(1,2,0).detach().cpu().numpy()
            gt = label_batch.squeeze(0).permute(1,2,0).detach().cpu().numpy()

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_min = np.min(img)
            img_max = np.max(img)
            img = 255 * (img - img_min) / (img_max - img_min)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img_file = str(i_batch) + "_img.jpg"
            prd_file = str(i_batch) + "_pred.jpg"
            gt_file = str(i_batch) + "_gt.jpg"


            os.makedirs(test_save_path + '/' + str(i_batch) + '/', exist_ok=True)

            cv2.imwrite(test_save_path + '/' + str(i_batch) + '/' + img_file, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(test_save_path + '/' + str(i_batch) + '/' + prd_file, prd*255)
            cv2.imwrite(test_save_path + '/' + str(i_batch) + '/' + gt_file, gt*255)


        metriclist += np.array(metric_list)
        logging.info('idx %d mean_dice %f mean_jc %f' % (i_batch, np.mean(metric_list, axis=0)[0], np.mean(metric_list, axis=0)[1]))
    
    metriclist = metriclist / len(testloader)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_jc %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    
    performance = np.mean(metriclist, axis=0)[0]
    mean_jc = np.mean(metriclist, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_jc : %f' % (performance, mean_jc))
    return "Testing Finished!"



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

    dataset_config = {
        'Polyp': {
          'Dataset': PolypDataset,
          'num_classes': 2,
          'img_root': '/content/drive/MyDrive/MyBScProject/project_TransUNet/data/Polyp/Original/',
          'gt_root': '/content/drive/MyDrive/MyBScProject/project_TransUNet/data/Polyp/Ground Truth/',
        },
        'Kvasir': {
          'Dataset': PolypDataset,
          'num_classes': 2,
          'img_root': '/content/drive/MyDrive/datasets/Kvasir-SEG/images/',
          'gt_root': '/content/drive/MyDrive/datasets/Kvasir-SEG/masks/',
        },
        'Ph2': {
          'Dataset': PolypDataset,
          'num_classes': 2,
          'img_root': '/content/drive/MyDrive/datasets/ph2/trainx/',
          'gt_root': '/content/drive/MyDrive/datasets/ph2/trainy/',
        },
        'CVCKvasir': {
          'Dataset': PolypDataset,
          'num_classes': 2,
          'img_root': '/content/drive/MyDrive/datasets/CVC-Kvasir/images/',
          'gt_root': '/content/drive/MyDrive/datasets/CVC-Kvasir/masks/',
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.img_root = dataset_config[dataset_name]['img_root']
    args.gt_root = dataset_config[dataset_name]['gt_root']
    args.is_pretrain = True
    args.exp = ('TU_' if args.arch == 'Transunet' else 'TransCASCADE_') + dataset_name + str(args.img_size)  # Both Transunet and TransCASCADE
    snapshot_path = "/content/drive/MyDrive/MyTransunet/TransUNet-repo/model/{}/{}".format(args.exp, ('TU' if args.arch == 'Transunet' else 'TransCASCADE'))  # Both Transunet and TransCASCADE
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))

    nets = {
        'Transunet': ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda(),
        'Transcascade': TransCASCADE(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    }
    net = nets[args.arch]  # Both Transunet and TransCASCADE
    net.load_from(weights=np.load(config_vit.pretrained_path))
    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))

    if dataset_name == 'Polyp':
        snapshot = '/content/drive/MyDrive/MyTransunet/TransUNet-repo/model/TU_Polyp224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs16_224/epoch_149.pth' # Transunet
        # snapshot = '/content/drive/MyDrive/MyTransunet/TransUNet-repo/model/TransCASCADE_Polyp224/TransCASCADE_pretrain_R50-ViT-B_16_skip3_epo150_bs16_lr0.0001_224/epoch_149.pth'  # TransCASCADE lr=0.01
        # snapshot = '/content/drive/MyDrive/MyTransunet/TransUNet-repo/model/TransCASCADE_Polyp224/TransCASCADE_pretrain_R50-ViT-B_16_skip3_epo150_bs16_224/epoch_149.pth'  # TransCASCADE
    elif dataset_name == 'Kvasir':
        snapshot = '/content/drive/MyDrive/MyTransunet/TransUNet-repo/model/TU_Kvasir224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs16_224/epoch_149.pth'
    elif dataset_name == 'Ph2':
        # snapshot = '/content/drive/MyDrive/MyTransunet/TransUNet-repo/model/TU_Ph2224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs16_224/epoch_149.pth'  # Transunet
        # snapshot = '/content/drive/MyDrive/MyTransunet/TransUNet-repo/model/TransCASCADE_Ph2224/TransCASCADE_pretrain_R50-ViT-B_16_skip3_epo150_bs16_224/epoch_149.pth'  # TransCASCADE lr=0.01
        snapshot = '/content/drive/MyDrive/MyTransunet/TransUNet-repo/model/TransCASCADE_Ph2224/TransCASCADE_pretrain_R50-ViT-B_16_skip3_epo150_bs16_lr0.0001_224/epoch_149.pth'  # TransCASCADE lr=0.0001
    elif dataset_name == 'CVCKvasir':
        snapshot = '/content/drive/MyDrive/MyTransunet/TransUNet-repo/model/TU_CVCKvasir224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs16_224/epoch_149.pth'  # Transunet
        # snapshot = '/content/drive/MyDrive/MyTransunet/TransUNet-repo/model/TransCASCADE_CVCKvasir224/TransCASCADE_pretrain_R50-ViT-B_16_skip3_epo150_bs16_lr0.0001_224/epoch_149.pth'  # TransCASCADE lr=0.0001

    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = '/content/drive/MyDrive/MyTransunet/TransUNet-repo/TransUNet/test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_save:
        if dataset_name == 'Polyp':
            test_save_path = f'/content/drive/MyDrive/MyTransunet/results/polyp'  # Transunet
            # test_save_path = f'/content/drive/MyDrive/MyTransunet/results/polyp-cascade'  # TransCASCADE
        elif dataset_name == 'Kvasir':
            test_save_path = f'/content/drive/MyDrive/MyTransunet/results/kvasir'
        elif dataset_name == 'Ph2':
            # test_save_path = f'/content/drive/MyDrive/MyTransunet/results/ph2'  # Transunet
            test_save_path = f'/content/drive/MyDrive/MyTransunet/results/ph2-cascade'  # TransCASCADE
        elif dataset_name == 'CVCKvasir':
            test_save_path = f'/content/drive/MyDrive/MyTransunet/results/cvckvasir'  # Transunet
            # test_save_path = f'/content/drive/MyDrive/MyTransunet/results/cvckvasir-cascade'  # TransCASCADE
    else:
        test_save_path = None
    inference(args, net, test_save_path)