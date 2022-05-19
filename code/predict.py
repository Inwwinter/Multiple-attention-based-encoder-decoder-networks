import torch
from torch.autograd import Variable
import numpy as np
from dataset import ListDataset
from model import make_model
from dataset import vocab, AttnLabelConverter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
characters = '0123456789'
max_len = 15
attn_converter = AttnLabelConverter(characters)
num_class = len(attn_converter.character)
def test(model):
    hit = 0
    all = 0
    # valid_path='D:/BaiduNetdiskDownload/data/Number_test/gunlun'
    valid_path='/data/ocr_dataset/Number_test/gunlun'
    dataloader = torch.utils.data.DataLoader(ListDataset(valid_path), batch_size=16, shuffle=False, num_workers=4, drop_last=False)
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size = imgs.size(0)
        img = imgs.to(device)
        length_for_pred = torch.IntTensor([max_len] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, max_len + 1).fill_(0).to(device)
        text_for_loss, length_for_loss = attn_converter.encode(labels, batch_max_length=max_len)
        
        _, probs = model(img, text_for_pred, False, max_len)

        probs = probs[:, :text_for_loss.shape[1] - 1, :]
        _, preds_index = probs.max(2)
        preds_str = attn_converter.decode(preds_index, length_for_pred)
        for pred, label in zip(preds_str, labels):
            all+=1
            pred = pred[:pred.find('[s]')]
            if pred!=label:
                hit += 1
    #             print('label:', label, 'pred:', pred, hit, all, hit/all)
    # print('accuracy = %.2f%%'%(100.0-100.0*hit/all))
    
    return 1.0-1.0*hit/all


if __name__ == '__main__':

    import time
    start=time.time()

    model = make_model(num_class)
    model.load_state_dict(torch.load('checkpoint/gunlun_02/best.pth'))
    model.cuda()
    model.eval()
    test(model)

    print((time.time()-start)/500.0)

