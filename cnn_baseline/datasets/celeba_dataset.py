import os
import io
import torch
import numpy as np
import pickle
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
WORKING_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

class CelebADataset(Dataset):
    def __init__(self, data_dir, split, attr_names, img_size=128, transform=None):
        super(CelebADataset, self).__init__()
        assert split in ['train', 'valid', 'test'], "Unkown split {}!".format(split)

        self.data_dir = data_dir
        self.attr_path = os.path.join(data_dir, 'Anno', 'list_attr_celeba.txt')
        self.split = split
        if len(attr_names) == 0:
            self.attr_names = [key for key in self.attr2idx.keys()]
        else:
            self.attr_names = attr_names
        self.img_size = img_size
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()
        self.box = (3, 26, 173, 196)

        # load dataset
        with open(os.path.join(self.data_dir, 'celeba_pkl', '{}.pkl'.format(self.split)), "rb") as f:
            self.dataset = pickle.load(f)
            f.close()


        if transform is None:
            raise ValueError("Do not support transform to be None")
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        byte_img, label = self.dataset[index]
        image = Image.open(io.BytesIO(byte_img))
        # crop image
        image = image.crop(self.box)
        return self.transform(image), torch.FloatTensor(label)

    def preprocess(self):
        """Preprocess the CelebA attribute file"""

        # check if pickel file already exist
        if os.path.exists(os.path.join(self.data_dir, 'celeba_pkl', 'train.pkl')):
            print('The CelebA dataset has been preprocessed....load them directly')
            return


        train_dataset = []
        valid_dataset = []
        test_dataset = []

        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        np.random.seed(1234)
        np.random.shuffle(lines)
        for i, line in tqdm(enumerate(lines)):
            split = line.split()
            filename = split[0]
            values = split[1:]

            # load data, convert to bytes and save in memory
            img = Image.open(os.path.join(self.data_dir, 'img_align_celeba', filename), mode='r')
            # img.show()
            byte_img = io.BytesIO()
            img.save(byte_img, format='PNG')
            byte_img = byte_img.getvalue()

            label = []
            for attr_name in self.attr_names:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i + 1) < 1000:
                test_dataset.append([byte_img, label])
            elif 1000 <= (i + 1) < 2000:
                valid_dataset.append([byte_img, label])
            else:
                train_dataset.append([byte_img, label])

        # dump dataset to pickle
        for dataset, name in zip([train_dataset, valid_dataset, test_dataset], ['train', 'valid', 'test']):
            file = open(os.path.join(self.data_dir, 'celeba_pkl', '{}.pkl'.format(name)), "wb")
            pickle.dump(dataset, file)
            file.close()

        print('Finished preprocessing the CelebA dataset...')

if __name__ == '__main__':
    # # the code are used for preprocessing
    # atts = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Male',
    #         'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
    # preprocessing(data_dir='/Users/leon/Projects/stgan_pytorch/data', split='val', atts=atts)

    # the code are used for check dataset working
    dataset = CelebADataset(data_dir='/Users/leon/Projects/stgan_pytorch/data', split='val', img_size=64)

    for i in range(len(dataset)):
        img, atts = dataset.__getitem__(i)
        img.show()
        print(atts)