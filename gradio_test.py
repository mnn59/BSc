import argparse
import torch
import cv2
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from datasets.dataset_polyp import PolypDataset
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
import gradio as gr



parser = argparse.ArgumentParser()

parser.add_argument('--img_root', type=str, default='/content/drive/MyDrive/MyBScProject/project_TransUNet/data/Polyp/Original/', help='root dir for data')
parser.add_argument('--gt_root', type=str, default='/content/drive/MyDrive/MyBScProject/project_TransUNet/data/Polyp/Ground Truth/', help='root dir for mask')
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--test_save_dir', type=str, default='/content/drive/MyDrive/MyTransunet/TransUNet-repo/predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()




if __name__ == "__main__":
    # Define the preprocessing transforms
    preprocess = transforms.Compose([
        transforms.RandomRotation(90, expand=False, center=None, fill=None),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trnsfrm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


    def predict(image_type, image):
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
              'gt_root': '/content/drive/MyDrive/datasets/CVC-Kvasir/masks/'
            },
        }

        dataset_name = ('CVCKvasir' if image_type == 'Polyp' else 'Ph2')

        args.num_classes = dataset_config[dataset_name]['num_classes']
        args.Dataset = dataset_config[dataset_name]['Dataset']
        args.img_root = dataset_config[dataset_name]['img_root']
        args.gt_root = dataset_config[dataset_name]['gt_root']
        args.is_pretrain = True

        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
        if args.vit_name.find('R50') !=-1:
            config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
        
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)

        if dataset_name == 'Polyp':
            snapshot = '/content/drive/MyDrive/MyTransunet/TransUNet-repo/model/TU_Polyp224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs16_224/epoch_149.pth'
        elif dataset_name == 'Kvasir':
            snapshot = '/content/drive/MyDrive/MyTransunet/TransUNet-repo/model/TU_Kvasir224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs16_224/epoch_149.pth'
        elif dataset_name == 'Ph2':
            snapshot = '/content/drive/MyDrive/MyTransunet/TransUNet-repo/model/TU_Ph2224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs16_224/epoch_149.pth'
            
        elif dataset_name == 'CVCKvasir':
            snapshot = '/content/drive/MyDrive/MyTransunet/TransUNet-repo/model/TU_CVCKvasir224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs16_224/epoch_149.pth'
            # snapshot = '/content/drive/MyDrive/MyTransunet/TransUNet-repo/model/TransCASCADE_Ph2224/TransCASCADE_pretrain_R50-ViT-B_16_skip3_epo150_bs16_lr0.0001_224/epoch_149.pth'
        net.load_state_dict(torch.load(snapshot))

        image = Image.fromarray(image)
        _image = trnsfrm(image)  # tensor
        _image = _image.unsqueeze(0)  # ([1, 3, 224, 224])
        with torch.no_grad():
              prediction = torch.argmax(torch.softmax(net(_image), dim=1), dim=1)
        mask = prediction.squeeze().numpy()
        mask = Image.fromarray((mask * 255).astype(np.uint8))
        return mask


    inputs = gr.inputs.Image()
    outputs = gr.outputs.Image(type='pil').style(width=224, height=224, margin='auto')
    gr.Interface(fn=predict, 
                inputs=[
                    gr.Radio(["Polyp", "Skin lesion"], label="Image Type", info="Choose type of medical image"),  # change 'Skin' to 'Skin Lesion', 'Datasets' to 'Image Type'
                    inputs,
                ], 
                outputs=outputs,
                flagging_options=["correct ✅", "incorrect ❌", "other"],
                capture_session=True,
                examples=[
                    ['Skin lesion',  os.path.join(os.path.abspath(''), "/content/drive/MyDrive/datasets/ph2/trainx/X_img_0.bmp")],
                    ['Polyp',  os.path.abspath('../../../../../gr-exmpls/123.png')],
                ], 
                title='Medical Image Segmentation System (MISS)',
                description='This is image segmentation system for medical images of polyps and skin lesions. This app is implemented by `@Mahdi_Niknejad`'
                ).launch(share=True, debug=True)