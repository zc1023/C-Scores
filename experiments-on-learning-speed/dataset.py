
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
import torch 
import numpy as np

class SplitDataset(Dataset):
    def __init__(self,datalist,transforms=None):
        super(SplitDataset,self).__init__()
        self.transforms = transforms
        self.datalist = datalist
    def __getitem__(self, index):
        data = self.datalist[index]
        if self.transforms:
            data[0] = self.transforms(data[0])
        # print(data)
        return data
    def __len__(self):
        return(len(self.datalist))

cscores = np.load('cifar100-cscores-orig-order.npz')
cscorestest = np.load('cscorestest.npy')

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
trainset = datasets.CIFAR100(root='./CIFAR-100', train=True, download=True)
testset = datasets.CIFAR100(root='./CIFAR-100', train=False, download=True)

# Decorete the dataset
def data_split(rawset):

    return np.array(rawset)

trainset = data_split(trainset)
testset = data_split(testset)

def get_subset(dataset,cscore):
    # parament:csore
    def divide(cscore):
        # return:index
        subset_ratios = np.linspace(0, 1, 21)
        for i in range(len(subset_ratios)-1):
            if (cscore[1]> round(subset_ratios[i],4)) and cscore[1] <= round(subset_ratios[i+1],4) or (i==0 and cscore[1]==0) :
                return f"{round(subset_ratios[i],4)}--{round(subset_ratios[i+1],4)}"
    from itertools import groupby
    cscore = sorted(enumerate(cscore),key=lambda x:x[1])
    group_index = groupby(cscore,key=divide)
    subsets = []
    for k,indexs in group_index:
        a = np.stack(i[0] for i in list(indexs))
        subdataset = dataset[a]
        subsets.append({"key":k,'dataset':SplitDataset(subdataset.tolist(),transforms=transform_test)}) 
    return subsets

trainsubset = get_subset(trainset,cscores['scores'])
testsubset = get_subset(testset,cscorestest)
trainset = SplitDataset(trainset.tolist(),transforms=transform_train)
testset = SplitDataset(testset.tolist(),transforms=transform_test)

if __name__ == "__main__":
    testloader = DataLoader(testset,batch_size=128,shuffle=False, num_workers=2)
    for data in testloader:
        print(data)
    
