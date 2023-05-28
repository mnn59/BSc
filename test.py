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
# from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_polyp import PolypDataset
from utils import test_single_volume, calculate_metric_percase
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from datasets.dataset_polyp import test_dataset, get_loader

parser = argparse.ArgumentParser()

parser.add_argument('--img_root', type=str,
                    default='/content/drive/MyDrive/MyBScProject/project_TransUNet/data/Polyp/Original/', help='root dir for data')
parser.add_argument('--gt_root', type=str,
                    default='/content/drive/MyDrive/MyBScProject/project_TransUNet/data/Polyp/Ground Truth/', help='root dir for mask')

# parser.add_argument('--volume_path', type=str,
#                     default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--volume_path', type=str,
                    default='/content/drive/MyDrive/MyTransunet/TransUNet-repo/data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Polyp', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
# parser.add_argument('--list_dir', type=str,
#                     default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--list_dir', type=str,
                    default='/content/drive/MyDrive/MyTransunet/TransUNet-repo/TransUNet/lists/lists_Synapse', help='list dir')
parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
# parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--is_save', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='ViT-B_16', help='select one vit model')

# parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--test_save_dir', type=str, default='/content/drive/MyDrive/MyTransunet/TransUNet-repo/predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    # db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    # testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    # testloader = get_loader(args.img_root, args.gt_root, args.img_size)
    testloader = get_loader(args.img_root, args.gt_root, split="test", batchsize=1, trainsize=args.img_size, augmentation=False)

    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metriclist = 0.0
    for i_batch, sampled_batch in enumerate(tqdm(testloader)):
        # h, w = sampled_batch["image"].size()[2:]
        # image, label = sampled_batch[0], sampled_batch[1]

        # image_batch, label_batch = sampled_batch[0], sampled_batch[1]
        image_batch, label_batch = sampled_batch[0], sampled_batch[1]


        # print(image)
        # print(label)
        # print(type(image))  # tensor
        # input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        # input = image.unsqueeze(0).unsqueeze(0).float().cuda()
        # model.eval()

        # print('input: ', input)
        # print('input_shape', input.shape)
        # print('input_type', type(input))
        model.eval()

        with torch.no_grad():
            # prediction = torch.argmax(torch.softmax(model(image_batch), dim=1), dim=1)
            # print("hereeeeee")
            # prediction = torch.argmax(torch.softmax(model(image_batch.to(torch.device('cuda:0'))), dim=1), dim=1)
            image_batch = image_batch.float().cuda()
            # .detach().cpu().numpy()
            prediction = torch.argmax(torch.softmax(model(image_batch), dim=1), dim=1)
            # print(type(image_batch))
            # print("gereeeeee")
            # print(prediction.shape)
        metric_list = []
        for i in range(1, args.num_classes):
            # metric_list.append(calculate_metric_percase(prediction.numpy() == i, label_batch.numpy() == i))
            metric_list.append(calculate_metric_percase(prediction.detach().cpu().numpy() == i, label_batch.detach().cpu().numpy() == i))
            # metric_list.append(calculate_metric_percase(prediction.numpy() == i, label_batch.numpy() == i))
        
        if test_save_path is not None:
            # x.numpy().astype(np.float32)

            img = image_batch.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            prd = prediction.permute(1,2,0).detach().cpu().numpy()
            gt = label_batch.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            
            # visualize(original_image=img,
            #           mask_of_image=gt)

            # img_itk = sitk.GetImageFromArray(image_batch.detach().cpu().numpy().astype(np.float32))
            # prd_itk = sitk.GetImageFromArray(prediction.detach().cpu().numpy().astype(np.float32))
            # lab_itk = sitk.GetImageFromArray(label_batch.detach().cpu().numpy().astype(np.float32))
            # img_itk.SetSpacing((1, 1, 1))
            # prd_itk.SetSpacing((1, 1, 1))
            # lab_itk.SetSpacing((1, 1, 1))
            # sitk.WriteImage(prd_itk, test_save_path + '/' + str(i_batch) + "_pred.nii.gz")
            # sitk.WriteImage(img_itk, test_save_path + '/' + str(i_batch) + "_img.nii.gz")
            # sitk.WriteImage(lab_itk, test_save_path + '/' + str(i_batch) + "_gt.nii.gz")



            # this is test_save_path
            # _path = f'/content/drive/MyDrive/MyTransunet/results/polyp' + '/' + str(i_batch) + '/'
            

            img_file = str(i_batch) + "_img.jpg"
            prd_file = str(i_batch) + "_pred.jpg"
            gt_file = str(i_batch) + "_gt.jpg"


            os.makedirs(test_save_path + '/' + str(i_batch) + '/', exist_ok=True)

            cv2.imwrite(test_save_path + '/' + str(i_batch) + '/' + img_file, cv2.cvtColor(img*255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(test_save_path + '/' + str(i_batch) + '/' + prd_file, prd*255)
            cv2.imwrite(test_save_path + '/' + str(i_batch) + '/' + gt_file, gt*255)


    
    # print(metric_list)

        metriclist += np.array(metric_list)
        # logging.info('idx %d mean_dice %f mean_hd95 %f' % (i_batch, np.mean(metric_list, axis=0)[0], np.mean(metric_list, axis=0)[1]))
        logging.info('idx %d mean_dice %f mean_jc %f' % (i_batch, np.mean(metric_list, axis=0)[0], np.mean(metric_list, axis=0)[1]))
    
    metriclist = metriclist / len(testloader)
    # print(len(testloader))
    for i in range(1, args.num_classes):
        # logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
        logging.info('Mean class %d mean_dice %f mean_jc %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    
    performance = np.mean(metriclist, axis=0)[0]
    # mean_hd95 = np.mean(metriclist, axis=0)[1]
    mean_jc = np.mean(metriclist, axis=0)[1]
    # logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    logging.info('Testing performance in best val model: mean_dice : %f mean_jc : %f' % (performance, mean_jc))
    





    #     metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
    #                                   test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
    #     metric_list += np.array(metric_i)
    #     logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    # metric_list = metric_list / len(db_test)
    # for i in range(1, args.num_classes):
    #     logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    # performance = np.mean(metric_list, axis=0)[0]
    # mean_hd95 = np.mean(metric_list, axis=0)[1]
    # logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
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
        }
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    # args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    # args.list_dir = dataset_config[dataset_name]['list_dir']
    # args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.img_root = dataset_config[dataset_name]['img_root']
    args.gt_root = dataset_config[dataset_name]['gt_root']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    # snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = "/content/drive/MyDrive/MyTransunet/TransUNet-repo/model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
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
    # net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))

    # new
    # snapshot = '/content/drive/MyDrive/MyTransunet/TransUNet-repo/model/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth'
    if dataset_name == 'Polyp':
        snapshot = '/content/drive/MyDrive/MyTransunet/TransUNet-repo/model/TU_Polyp224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs16_224/epoch_149.pth'
    elif dataset_name == 'Kvasir':
        snapshot = '/content/drive/MyDrive/MyTransunet/TransUNet-repo/model/TU_Kvasir224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs16_224/epoch_149.pth'
    elif dataset_name == 'Ph2':
        snapshot = '/content/drive/MyDrive/MyTransunet/TransUNet-repo/model/TU_Ph2224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs16_224/epoch_149.pth'

    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    # log_folder = './test_log/test_log_' + args.exp
    log_folder = '/content/drive/MyDrive/MyTransunet/TransUNet-repo/TransUNet/test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    # if args.is_savenii:
    if args.is_save:
        print("inja")
        # args.test_save_dir = '../predictions'
        # args.test_save_dir = '/content/drive/MyDrive/MyTransunet/TransUNet-repo/predictions'
        # test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        # os.makedirs(test_save_path, exist_ok=True)
        if dataset_name == 'Polyp':
            test_save_path = f'/content/drive/MyDrive/MyTransunet/results/polyp'
        elif dataset_name == 'Kvasir':
            test_save_path = f'/content/drive/MyDrive/MyTransunet/results/kvasir'
        elif dataset_name == 'Ph2':
            test_save_path = f'/content/drive/MyDrive/MyTransunet/results/ph2'
    else:
        test_save_path = None
    inference(args, net, test_save_path)


