from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
import numpy as np
import torch


class GroupbyCIFARDataset(Dataset):
    def __init__(self,trainsubset):
        self.trainsubset = trainsubset
    
    def __len__(self):
        return len(self.trainsubset)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.trainsubset[idx]['data']
        label = self.trainsubset[idx]['label']

        sample = {'data':data,'label':label}
        return sample



cscores = np.load('cifar10-cscores-orig-order.npz')
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./CIFAR', train=True, download=True, transform=transform)

#build all trainsset
transloader = DataLoader(trainset,batch_size=32,shuffle=True, num_workers=2)

trainset_total = []


# Split the training set by C-Scores
for i,datas in enumerate(trainset):
    datas_dic = {}
    datas_dic['data']=datas[0]
    datas_dic['label']=datas[1]
    datas_dic['cscores']=cscores['scores'][i]
    trainset_total.append(datas_dic)


def g(x):
    subset_ratios = np.linspace(0, 0.95, 20)
    for i in range(len(subset_ratios)-1):
        if (x['cscores']>round(subset_ratios[i],2)) and x['cscores']<=round(subset_ratios[i+1],2):
            return f"{round(subset_ratios[i],2)}--{round(subset_ratios[i+1],2)}"

from itertools import groupby
train_sort = sorted(trainset_total,key = lambda x:x['cscores'])
train_group = groupby(train_sort,key = g)
keys = []
trainsgroup=[]
for key,group in train_group:
    keys.append(key)
    trainsgroup.append(list(group))

#build trainsdatasets
trainsdatalaoders = []
for i in range(len(trainsgroup)):
    trainsdatalaoders.append({'key':keys[i],'dataloader':DataLoader(GroupbyCIFARDataset(trainsgroup[i]),batch_size=32,shuffle=True, num_workers=2)})



if __name__ == "__main__":
    print(f"dataloader length {len(trainsdatalaoders)}")