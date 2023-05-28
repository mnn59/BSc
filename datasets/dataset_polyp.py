# import os
# import random

# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import torch
# from scipy import ndimage
# from scipy.ndimage.interpolation import zoom
# from torch.utils.data import Dataset
# from torchvision import transforms



# def random_rot_flip(image, mask):
#     k = np.random.randint(0, 4)
#     image = np.rot90(image, k)  # rotate
#     mask = np.rot90(mask, k)
#     axis = np.random.randint(0, 2)
#     image = np.flip(image, axis=axis).copy()  # flip
#     mask = np.flip(mask, axis=axis).copy()
#     return image, mask


# def random_rotate(image, mask):
#     angle = np.random.randint(-20, 20)
#     image = ndimage.rotate(image, angle, order=0, reshape=False)
#     mask = ndimage.rotate(mask, angle, order=0, reshape=False)
#     return image, mask



# # def flip_horizontal(image, mask):
# #     image = np.flip(image, axis=1).copy()
# #     mask = np.flip(mask, axis=1).copy()
# #     return image, mask
# #
# #
# # def rotate(image, mask, angle_abs=5):
# #     h, w, _ = image.shape
# #     angle = random.choice([angle_abs, -angle_abs])
# #
# #     M = cv2.getRotationMatrix2D((h, w), angle, 1.0)
# #     image = cv2.warpAffine(image, M, (h, w), flags=cv2.INTER_CUBIC)
# #     mask = cv2.warpAffine(mask, M, (h, w), flags=cv2.INTER_CUBIC)
# #     mask = np.expand_dims(mask, axis=-1)
# #     return image, mask


# # handler of transforms
# # class RandomGenerator(object):
# #     def __init__(self, output_size):
# #         self.output_size = output_size

# #     def __call__(self, sample):
# #         image, mask = sample['image'], sample['mask']

# #         if random.random() > 0.5:
# #             image, mask = random_rot_flip(image, mask)
# #         elif random.random() > 0.5:
# #             image, mask = random_rotate(image, mask)

# #         print(image.shape)
# #         x, y = image.shape
# #         if x != self.output_size[0] or y != self.output_size[1]:
# #             image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why 3?
# #             mask = zoom(mask, (self.output_size[0] / x, self.output_size[1] / y), order=0)
# #         image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
# #         mask = torch.from_numpy(mask.astype(np.float32))
# #         sample = {'image': image, 'mask': mask.long()}  # why long?
# #         return sample


# class RandomGenerator:
#     augmentations = [random_rot_flip, random_rotate]
#     # augmentations = [flip_horizontal, rotate]

#     def __init__(self, max_augment_count):
#         if max_augment_count <= len(self.augmentations):
#             self.max_augment_count = max_augment_count
#         else:
#             self.max_augment_count = len(self.augmentations)

#     def __call__(self, sample):
#         image, mask = sample['image'], sample['mask']

#         augmentation_count = random.randint(0, self.max_augment_count)
#         selected_augmentations = random.sample(self.augmentations, k=augmentation_count)
#         for augmentation in selected_augmentations:
#             image, mask = augmentation(image, mask)

#         return {'image': image, 'mask': mask}


# class RandomGenerator:
#     augmentations = [random_rot_flip, random_rotate]

#     def __init__(self, output_size):
#         self.output_size = output_size

#     def __call__(self, sample):
#         image, mask = sample['image'], sample['mask']

#         y, x, _ = image.shape
        
#         if x != self.output_size[0] or y != self.output_size[1]:
#             image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why 3?
#             mask = zoom(mask, (self.output_size[0] / x, self.output_size[1] / y), order=0)

#         augmentation_count = random.randint(0, len(self.augmentations))
#         # augmentation_count = 0
#         selected_augmentations = random.sample(self.augmentations, k=augmentation_count)
#         print(selected_augmentations)
#         for augmentation in selected_augmentations:
#             image, mask = augmentation(image, mask)

#         return {'image': image, 'mask': mask}


# def dataset_helper(base_dir):
#     metadata_df = pd.read_csv(os.path.join(base_dir, 'metadata.csv'))
#     metadata_df = metadata_df[['frame_id', 'png_image_path', 'png_mask_path']]
#     metadata_df['png_image_path'] = metadata_df['png_image_path'].apply(
#         lambda img_pth: os.path.join(base_dir, '/'.join(img_pth.split('/')[1:])))
#     metadata_df['png_mask_path'] = metadata_df['png_mask_path'].apply(
#         lambda img_pth: os.path.join(base_dir, '/'.join(img_pth.split('/')[1:])))
#     # Shuffle DataFrame
#     metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
#     test_df = metadata_df.sample(frac=0.1, random_state=42)
#     train_df = metadata_df.drop(test_df.index)
#     return train_df, test_df


# # no usage
# # def class_helper(base_dir):
# #     class_dict = pd.read_csv(os.path.join(base_dir, 'class_dict.csv'))
# #     # Get class names
# #     class_names = class_dict['class_names'].tolist()
# #     # Get class RGB values
# #     class_rgb_values = class_dict[['r', 'g', 'b']].values.tolist()
# #     return class_names, class_rgb_values


# def one_hot_encode(label, label_values):
#     semantic_map = []
#     for color in label_values:
#         equality = np.equal(label, color)
#         class_map = np.all(equality, axis=-1)
#         semantic_map.append(class_map)
#     semantic_map = np.stack(semantic_map, axis=-1)
#     return semantic_map


# def reverse_one_hot(image):
#     x = np.argmax(image, axis=-1)
#     return x


# def color_code_segmentation(image, label_values):
#     color_codes = np.array(label_values)
#     x = color_codes[image.astype(int)]
#     return x


# def visualize(**images):
#     '''plot images in one row'''
#     # n_images = len(images)
#     # plt.figure(figsize=(20, 8))
#     # for idx, (name, image) in enumerate(images.items()):
#     #     plt.subplot(1, n_images, idx + 1)
#     #     plt.xticks([])
#     #     plt.yticks([])
#     #     plt.title(name.replace('_', ' ').title(), fontsize=20)
#     #     plt.imshow(image)
#     plt.imshow(images)
#     plt.show()


# class Polyp_dataset(Dataset):
#     def __init__(self, base_dir, split, transform=None):
#         # super().__init__()
#         if split == "train":
#             df = dataset_helper(base_dir)[0]
#         else:
#             df = dataset_helper(base_dir)[1]
#         self.image_paths = df['png_image_path'].tolist()
#         self.mask_paths = df['png_mask_path'].tolist()
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, index):
#         # read images and masks
#         image = cv2.cvtColor(cv2.imread(self.image_paths[index]), cv2.COLOR_BGR2RGB)
#         mask = cv2.cvtColor(cv2.imread(self.mask_paths[index]), cv2.COLOR_BGR2RGB)

#         # one-hot-encode the mask
#         # mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
#         mask = one_hot_encode(mask, [[0, 0, 0], [255, 255, 255]]).astype('float')

#         sample = {'image': image, 'mask': mask}

#         if self.transform:
#             sample = self.transform(sample)
#             image, mask = sample['image'], sample['mask']

#         return {'image': image, 'mask': mask}
#         # return image, mask
#         # return sample


# from google.colab.patches import cv2_imshow

# # Just for debug
# if __name__ == '__main__':

#     db_train = Polyp_dataset(
#         base_dir='/content/drive/MyDrive/MyBScProject/project_TransUNet/data/Polyp/',
#         split='train',
#         transform=transforms.Compose(
#             # [RandomGenerator(output_size=[224, 224])]
#             [RandomGenerator(2)]
#         )
#         # transform=None
#     )
#     print("The length of train set is: {}".format(len(db_train)))

#     # random_idx = random.randint(0, len(db_train) - 1)
#     image, mask = db_train[2]['image'], db_train[2]['mask']

#     select_class_rgb_values = np.array([[0, 0, 0], [255, 255, 255]])
    
    
#     # print(db_train[0][0].shape)

#     print(db_train[2]['image'].shape)
#     print(db_train[2]['mask'].shape)


#     # plt.imshow(db_train[2]['image'])
#     cv2_imshow(image)

#     # fig, ax = plt.subplots(1,2)
#     # ax[0].imshow(image)
#     # ax[1].imshow(color_code_segmentation(reverse_one_hot(mask), select_class_rgb_values))
#     # plt.show()

#     # visualize(
#     #     original_image=image,
#     #     ground_truth_mask=color_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
#     #     one_hot_encoded_mask=reverse_one_hot(mask)
#     # )








import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch



class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, split, trainsize, augmentations):
        self.trainsize = trainsize
        self.augmentations = augmentations
        # print(self.augmentations)
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.split = split
        split_size = .8  # train test split
        self.splt = int(np.floor(split_size * self.size))
        if self.split == "train":
          self.size = len(self.images[:self.splt])
        else:
          self.size = len(self.images[self.splt:])
        
        if self.augmentations == True:
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                # transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomRotation(90, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            

    def __getitem__(self, index):
        if self.split == 'train':
          image = self.rgb_loader(self.images[:self.splt][index])
          gt = self.binary_loader(self.gts[:self.splt][index])
        else: # also never called? no :)
          image = self.rgb_loader(self.images[self.splt:][index])
          gt = self.binary_loader(self.gts[self.splt:][index])
        
        # image = self.rgb_loader(self.images[index])
        # gt = self.binary_loader(self.gts[index])
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    
    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, split, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, augmentation=False):

    dataset = PolypDataset(image_root, gt_root, split, trainsize, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.splt:][self.index])
        gt = self.binary_loader(self.gts[self.splt:][self.index])
        # image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        # gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')






