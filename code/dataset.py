import torch
from torch._C import dtype
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable
from PIL import Image
import os
import cv2
import numpy as np
from torchvision import datasets, models, transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
imgW=400
imgH=64
max_len = 15
vocab =  "0123456789"

def get_images(path):
    return path
def get_labels(path):
    labels = []
    for label in path:
        name = os.path.basename(label)
        name = name.split('_')[0]
        labels.append(name)
    return labels

def illegal(label):
    if len(label) > max_len-1:
        return True
    for l in label:
        if l not in vocab[1:-1]:
            return True
    return False


class ListDataset(Dataset):
    def __init__(self, root, imgW=imgW, imgH=imgH):
        if os.path.isfile(root):
            self.images_path=[root]
        else:
            self.images_path = [os.path.join(root, img_name)
                            for img_name in os.listdir(root)]
        self.images = get_images(self.images_path)
        self.labels = get_labels(self.images_path)
        self.imgW=imgW
        self.imgH=imgH
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img_path, label = self.images[index], self.labels[index]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img) / 255.
        img = cv2.resize(img,(self.imgW,self.imgH))
        img = np.transpose(img,(2, 0, 1))
        # As pytorch tensor
        img = torch.from_numpy(img).float()
        return img, label

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, name):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.name = name
    def forward(self, x):
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name is self.name:
                b = x.size(0)
                c = x.size(1)
                return x.view(b, c, -1).permute(0, 2, 1)
        return None

class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]'] 
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=15):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts

class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['-'] + dict_character  # dummy '-' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=36):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            try:
               batch_text[i][:len(text)] = torch.LongTensor(text)
            except:
               print(text)
        return (batch_text.cuda(), torch.IntTensor(length).cuda())
        
if __name__=='__main__':
    train_path=r'D:\BaiduNetdiskDownload\data\plate\Number_test\gunlun'
    listdataset = ListDataset(train_path)
    dataloader = torch.utils.data.DataLoader(listdataset, batch_size=2, shuffle=False, num_workers=0)
    for epoch in range(1):
        for batch_i, (imgs, labels_y, labels) in enumerate(dataloader):
            print(labels_y,labels)
            break

