import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from skimage import io
import numpy as np

class pie_dataset(Dataset):
    def __init__(self, image_path, transform, mode, args):
        self.image_path = image_path
        self.transform = transform
        self.mode = mode
        self.lines = open("data/"+mode+".txt", 'r').readlines()
        # self.num_data = int(self.lines[0])
        self.args = args

        self.attr2idx = {}
        self.idx2attr = {}

        print ('Start preprocessing dataset..!')
        self.preprocess()
        print ('Finished preprocessing dataset..!')

        self.num_data = len(self.filenames_path)
        print("len of data:" + str(self.num_data))

    def preprocess(self):
        attrs = self.lines[1].split()
        for i, attr in enumerate(attrs):
            self.attr2idx[attr] = i
            self.idx2attr[i] = attr

        self.selected_attrs = ['case'+"%.2d" %case for case in range(20)]
        # self.selected_attrs = ['case'+"%.2d" %case for case in [0, 2, 4, 7, 9, 11, 13]]
        # self.selected_attrs = ['case'+"%.2d" %case for case in [1, 3, 5, 6, 8, 10, 12, 14, 15, 16, 17, 18, 19]]
        
        self.filenames_path = []
        self.filenames_label = []
        self.filenames_identity = []

        lines = self.lines[2:]
        if self.mode == 'train':
            random.shuffle(lines)   # random shuffling

        for i, line in enumerate(lines):

            splits = line.split()
            filename = splits[0]
            
            label = []
            if self.mode == 'train':

                identity = splits[2]
                values = splits[3:-2] # illu label
                
                for idx, value in enumerate(values):
                    attr = self.idx2attr[idx]
                    if attr in self.selected_attrs:
                        if value == '1':
                            label.append(1)
                        else:
                            label.append(0)

                self.filenames_label.append(label)
                self.filenames_identity.append(identity)

                fname_triplet = []
                fname_triplet.append(filename)
                fname_triplet.append(splits[-2])
                fname_triplet.append(splits[-1])
            else:
                fname_triplet = [filename]

            self.filenames_path.append(fname_triplet)
  
    def __getitem__(self, index):

        if not self.args.use_si:
            num_read_img = 1 # only me
        else:
            num_read_img = 3 # me, otherme and otherone

        images = []
        out_images = []
        fname_triplet = self.filenames_path[index]

        for i, fname in enumerate(fname_triplet):

            if i >= num_read_img:
                break

            if self.args.log_space:
                image_ = io.imread(os.path.join(self.image_path, fname)).astype("float32")
                image_ = (np.log(image_+1.0)/np.log(256.0)*255.0)
                image = Image.fromarray(np.uint8(image_))
            else:
                image = Image.open(os.path.join(self.image_path, fname))
            images.append(image)

        if self.mode == 'train':
            label = self.filenames_label[index]
            identity = self.filenames_identity[index]
            identity = [int(identity)]
  
            seed = np.random.randint(2147483647) # make a seed with numpy generator
            for img in images:
                random.seed(seed)
                out_images.append(self.transform(img))

            return out_images, torch.FloatTensor(label), torch.LongTensor(identity)
        else:
            return [self.transform(images[0])], self.filenames_path[index][0]

    def __len__(self):
        return self.num_data

def get_loader(image_path, crop_size, image_size, batch_size, dataset, args, mode='train'):
    """Build and return data loader."""
    osize = [image_size, image_size]
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize(osize, Image.ANTIALIAS),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            transforms.Resize(osize, Image.ANTIALIAS),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = pie_dataset(image_path, transform, mode, args)
    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             num_workers=4,
                             shuffle=shuffle)

    return data_loader