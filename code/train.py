from math import atan
import torch
import torch.nn as nn
from torch.autograd import Variable
import time

from torch.nn.functional import batch_norm
from dataset import ListDataset
from model import make_model
from predict import test
import random
import os
import numpy as np
from dataset import CTCLabelConverter, AttnLabelConverter
characters='0123456789'
max_len=15
best_acc = 0.0
attn_converter = AttnLabelConverter(characters)
ctc_converter = CTCLabelConverter(characters)
num_class = len(attn_converter.character)
filename='acc.txt'

def seed_torch(seed=1219):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def _init_fn(worker_id, seed=1219):
    np.random.seed(int(seed)+worker_id)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))


class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, criterion, opt=None):
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, loss_ctc):
        
        loss_attn = self.criterion(x.view(-1, x.shape[-1]), y.contiguous().view(-1))
        a = 0.8
        loss = a * loss_ctc + (1-a) * loss_attn
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data

def run_epoch(dataloader, model, loss_compute, ctc_loss):
    "Standard Training and Logging Function"
    start = time.time()
    for i, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.cuda()
        batch_size = imgs.size(0)
        attn_text, attn_length = attn_converter.encode(labels, max_len)
        target = attn_text[:, 1:]  # without < Symbol

        # 计算CTC损失
        ctc_text, ctc_length= ctc_converter.encode(labels, max_len)
        memory, out = model(imgs, attn_text[:, :-1], True, max_len)
        memory_size = torch.IntTensor([memory.size(1)] * batch_size)
        memory = memory.log_softmax(2).permute(1, 0, 2)
        loss_ctc = ctc_loss(memory, ctc_text, memory_size, ctc_length)
        
        loss = loss_compute(out, target, loss_ctc)
        if i % 50 == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss, batch_size / elapsed))
            file_handle = open('loss.txt', mode='a+')
            file_handle.write(str(loss.data)+' \n')
            file_handle.close()
            start = time.time()
            tokens = 0
        if i%400==0:
            model.eval()
            cur_acc = test(model)
            print("cur_acc:", cur_acc)
            file_handle = open('acc.txt', mode='a+')
            file_handle.write(str(cur_acc)+' \n')
            file_handle.close()
            global best_acc
            if best_acc < cur_acc:
                best_acc = cur_acc
                torch.save(model.state_dict(), 'checkpoint/best.pth')
            else:
                torch.save(model.state_dict(), 'checkpoint/latest.pth')
            model.train()
    return None


def train():
    seed_torch()
    batch_size = 64
    # train_path='D:/BaiduNetdiskDownload/data/Number_test/gunlun'
    # valid_path='D:/BaiduNetdiskDownload/data/Number_test/gunlun'
    train_path='/data/ocr_dataset/Number_train/gl'
    valid_path='/data/ocr_dataset/Number_test/gunlun'
    train_dataloader = torch.utils.data.DataLoader(ListDataset(train_path), batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False, pin_memory=True,
    worker_init_fn=_init_fn)
    val_dataloader = torch.utils.data.DataLoader(ListDataset(valid_path), batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True)
    model = make_model(num_class)
    # model.load_state_dict(torch.load('checkpoint/shumaguan/latest.pth'))
    model.cuda()
    
    model_opt = NoamOpt(model.tgt_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    ctc_loss = torch.nn.CTCLoss(zero_infinity=True).cuda()
    attn_loss = torch.nn.CrossEntropyLoss(ignore_index=0).cuda()
    for epoch in range(6):
        model.train()
        run_epoch(train_dataloader, model, 
              SimpleLossCompute(attn_loss, model_opt), ctc_loss)
        
if __name__=='__main__':
    train()
