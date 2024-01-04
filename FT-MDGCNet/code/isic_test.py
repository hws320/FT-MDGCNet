import os
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import time
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
#from pyramid_vig import pvig_ti_224_gelu, pvig_s_224_gelu, pvig_m_224_gelu, pvig_b_224_gelu
from pyramid_vig_3d import pvig_ti_224_gelu


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   
    data_transform = {
        "val": transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   #normalize
                                   ])}

    val_data = 'datasets/HAM(82)/test' 

    validate_dataset = datasets.ImageFolder(val_data,
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  shuffle=True,
                                                  num_workers=16)

    print("{} images for test.".format(val_num))

    print('Using {} dataloader workers every process'.format(16))
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = validate_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
 

    net = pvig_ti_224_gelu(num_classes=7)
    # load pretrain weights

    model_weight_path = "HAM10000.pth"
    net.load_state_dict(torch.load(model_weight_path))
    
    net.to(device)


    val_steps = len(validate_loader)


    acc_val = []
  
    net.eval()
    auc = 0.0
    acccc = 0.0
    acc = 0.0  # accumulate accurate number / epoch
    precision=0.0
    recall=0.0
    F1_score=0.0
    predlist=[]
    scorelist=[]
    targetlist=[]
    # labels = [0,1,2]
    labels = [0,1,2,3,4,5,6]
    with torch.no_grad():
        val_bar = tqdm(validate_loader)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            val_lab = label_binarize(val_labels.cpu(), classes=labels)
            predict_y = torch.max(outputs, dim=1)[1]
            val_pre = label_binarize(predict_y.cpu(), classes=labels)
            # print(val_labels,predict_y.cpu())
            # exit()
            predlist=np.append(predlist, predict_y.cpu().numpy())
            targetlist=np.append(targetlist,val_labels)
        test_folder = './new/model_test/isic{}'.format(time.strftime('%m-%d', time.localtime()))
        if not os.path.exists(test_folder):
            os.mkdir(test_folder)
        with open(test_folder+'/targetlist{}.txt'.format(time.strftime('%m-%d %H:%M', time.localtime())),'a+') as f:        
            for tar in targetlist:
                f.write(str(tar)+'\n')
                f.close
        with open(test_folder+'/predlist{}.txt'.format(time.strftime('%m-%d %H:%M', time.localtime())),'a+') as f:        
            for pre in predlist:
                f.write(str(pre)+'\n')
                f.close 
        acc = accuracy_score(targetlist, predlist, normalize=True)
        F1 = f1_score(targetlist, predlist, average='macro',zero_division=1)
        precision = precision_score(targetlist, predlist,  labels=None, pos_label=1, average='macro', sample_weight=None, zero_division='warn')
        recall = recall_score(targetlist, predlist,  labels=None, pos_label=1, average='macro', sample_weight=None, zero_division='warn')
        val_lab = label_binarize(targetlist, classes=labels)
        val_pre = label_binarize(predlist, classes=labels)
        auc = roc_auc_score(val_lab, val_pre, average='macro', multi_class='ovo')
        print('acc',acc)
        print('F1',F1)
        print('precision',precision)
        print('recall',recall)
        print('auc',auc)
        with open(test_folder+'/metrics{}.txt'.format(time.strftime('%m-%d %H:%M', time.localtime())),'a+') as f:  
            f.write('acc:{}, F1:{}, precision:{}, recall:{}, auc:{}\n'.format(acc,F1,precision,recall,auc))



#############用于计算混淆矩阵的
        leibie = ['MEL','NV','BCC','AKIEC','BKL','DF','VASC']
        confusion_mat = confusion_matrix(targetlist, predlist)
        print(confusion_mat)


    print('Finished Test！')



if __name__ == '__main__':
    main()
