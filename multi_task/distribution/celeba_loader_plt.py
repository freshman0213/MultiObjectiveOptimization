import os
import torch
import numpy as np
import scipy.misc as m
import imageio
from PIL import Image
import re
import glob

from torch.utils import data
import matplotlib.pyplot as plt
from tqdm import tqdm


class CELEBA(data.Dataset):
    def __init__(self, params, root, split="train", is_transform=False, img_size=(32, 32), augmentations=None, subset=True):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes =  len(params['tasks'])
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([73.15835921, 82.90891754, 72.39239876]) # TODO(compute this mean)
        self.files = {}
        self.labels = {}

        self.label_file = self.root+"/Anno/list_attr_celeba.txt"
        label_map = {}
        with open(self.label_file, 'r') as l_file:
            labels = l_file.read().split('\n')[2:-1]
        for label_line in labels:
            f_name = re.sub('jpg', 'png', label_line.split(' ')[0])
            label_txt = list(map(lambda x:int(x), re.sub('-1','0',label_line).split()[1:self.n_classes+1])) # TODO: get right label
            label_map[f_name]=label_txt
        
        self.subset = set()
        with open(self.root+'/subset.txt') as f:
            lines = f.readlines()
            for line in lines:
                self.subset.add(line.rstrip('\n'))
        print(self.split)
        self.all_files = glob.glob(self.root+'/Img/img_align_celeba_png/*.png')
        with open(root+'//Eval/list_eval_partition.txt', 'r') as f:
            fl = f.read().split('\n')
            fl.pop()
            if 'train' in self.split:
                if subset: 
                    selected_files = list(filter(lambda x:x.split(' ')[0] in self.subset and x.split(' ')[1]=='0', fl))
                else: 
                    selected_files = list(filter(lambda x:x.split(' ')[1]=='0', fl))
            elif 'val' in self.split:
                if subset: 
                    selected_files =  list(filter(lambda x:x.split(' ')[0] in self.subset and x.split(' ')[1]=='1', fl))
                else: 
                    selected_files = list(filter(lambda x:x.split(' ')[1]=='1', fl))
            elif 'test' in self.split:
                selected_files =  list(filter(lambda x:x.split(' ')[1]=='2', fl))
            selected_file_names = list(map(lambda x:re.sub('jpg', 'png', x.split(' ')[0]), selected_files))
        
        base_path = '/'.join(self.all_files[0].split('/')[:-1])
        self.files[self.split] = list(map(lambda x: '/'.join([base_path, x]), set(map(lambda x:x.split('/')[-1], self.all_files)).intersection(set(selected_file_names))))
        self.labels[self.split] = list(map(lambda x: label_map[x], set(map(lambda x:x.split('/')[-1], self.all_files)).intersection(set(selected_file_names))))
        self.class_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
                                'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',      
                                'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',       
                                'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 
                                'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 
                                'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
        self.class_names = self.class_names[:self.n_classes]
        label_txt_0 = np.ones(self.n_classes) * len(self.labels[self.split])
        label_txt_1 = np.zeros(self.n_classes)
        for l in tqdm(self.labels[self.split]):
            label_txt_1 = label_txt_1 + l
        label_txt_0 = label_txt_0 - label_txt_1
        label_txt_0 /= len(self.labels[self.split])
        label_txt_1 /= len(self.labels[self.split])
        print(len(self.labels[self.split]))
        print(label_txt_0)
        print(label_txt_1)
        fig = plt.subplots(figsize =(10, 7))
        ind = np.arange(self.n_classes)  
        p1 = plt.bar(ind, label_txt_0, 0.35)
        p2 = plt.bar(ind, label_txt_1, 0.35,
                    bottom = label_txt_0,)
        
        is_subset = " (subset with %d imgs)"%(len(self.labels[self.split])) if subset else ""
        plt.ylabel('Labels')
        plt.title('Label Distribution of %s set%s'%(self.split, is_subset))
        plt.xticks(ind, ind)
        plt.yticks(np.arange(0, 1, 10))
        plt.legend((p1[0], p2[0]), ('zeros', 'ones'), loc=4)
        plt.show()

        if len(self.files[self.split]) < 2:
            raise Exception("No files for split=[%s] found in %s" % (self.split, self.root))

        print("Found %d %s images" % (len(self.files[self.split]), self.split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        label = self.labels[self.split][index]
        img = imageio.imread(img_path)

        if self.augmentations is not None:
            img = self.augmentations(np.array(img, dtype=np.uint8))

        if self.is_transform:
            img = self.transform_img(img)

        return [img] + label

    def transform_img(self, img):
        """transform
        Mean substraction, remap to [0,1], channel order transpose to make Torch happy
        """
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((self.img_size[0], self.img_size[1])).convert('RGB'))
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img

if __name__ == '__main__':
    import torchvision
    import matplotlib.pyplot as plt

    params = {'tasks': ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                      "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39"]}
    local_path = './PATH_FOR_CELEBA_DATASET'
    subset = False
    split='test'
    dst = CELEBA(params, local_path, is_transform=True, augmentations=None, subset=subset, split=split)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    

    # for i, data in enumerate(trainloader):
    #     imgs = imgs.numpy()[:, ::-1, :, :]
    #     imgs = np.transpose(imgs, [0,2,3,1])

    #     f, axarr = plt.subplots(bs,4)
    #     for j in range(bs):
    #         axarr[j][0].imshow(imgs[j])
    #         axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
    #         axarr[j][2].imshow(instances[j,0,:,:])
    #         axarr[j][3].imshow(instances[j,1,:,:])
    #     plt.show()
    #     a = raw_input()
    #     if a == 'ex':
    #         break
    #     else:
    #         plt.close()
