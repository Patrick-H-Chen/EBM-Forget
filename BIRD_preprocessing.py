import os
import os
import csv
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


f = os.listdir("/data/CUB_200_2011/images/")
data = []
labels = []
for ff in f:
    if(ff[0]=='.'):
        continue
    label = int(ff.split(".")[0])
    types = os.listdir("/data/CUB_200_2011/images/"+ff)
    for t in types:
        if(t[0]=='.'):
            continue
        img_dir = "/data/CUB_200_2011/images/"+ff+"/"+t
        image = Image.open(img_dir)
        img = np.asarray(image)
        if(len(img.shape) < 3):
            img = np.expand_dims(img,-1).repeat(3,2)
        seg = np.asarray(Image.open("/data/segmentations/"+ff+"/"+t.split(".")[0]+".png"))
        if(len(seg.shape) > 2):
            seg = seg[:,:,0]
        mask = np.zeros(seg.shape)
        idx = np.where(seg > 150.)
        mask[idx] = 1
        a = idx[0].min()
        b = idx[0].max()
        c = idx[1].min()
        d = idx[1].max()
        extract = np.uint8(np.einsum("whc,wh->whc",img,mask)[a:b,c:d,:])
        data.append(np.asarray(Image.fromarray(extract).resize([32,32])))
        labels.append(label)

        data = np.stack(data)
labels = np.stack(labels)

qq = []
for i in range(200):
    qq.append(np.where(labels==i+1)[0].shape[0])
qq = np.asarray(qq)
idx = np.where(qq == 60)[0][:100]

idx = np.where(qq == 60)[0][:100]
train = []
tl = []
test = []
testl= []
for ii in idx:
    iiii = np.where(labels==ii+1)[0]
    train.append(data[iiii][:50])
    tl.append([ii]*50)
    test.append(data[iiii][50:])
    testl.append([ii]*(len(data[iiii]) - 50))
train = np.vstack(train)
tl = np.vstack(tl)
test = np.vstack(test)
testl = np.hstack(testl)
np.save("/data/BIRDtrain",train)
np.save("/data/BIRDtrainlabel",tl)
np.save("/data/BIRDtest",test)
np.save("/data/BIRDtestlabel",testl)
