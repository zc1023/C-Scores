
from torchvision import datasets, transforms

import numpy as np


cscores = np.load('cifar100-cscores-orig-order.npz')
cscorestest = np.load('cscorestest.npy')

transform = transforms.Compose(
    [
        transforms.RandomCrop((32,32),padding=4),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

trainset = datasets.CIFAR100(root='./CIFAR-100', train=True, download=True, transform=transform)
testset = datasets.CIFAR100(root='./CIFAR-100', train=False, download=True, transform=transform)

# Decorete the dataset
def data_split(rawset):
    dataset=[]

    for data in rawset:
        dataset.append({"image":data[0],"label":data[1]})
    return np.array(dataset)
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
        subsets.append({"key":k,'dataset':subdataset}) 
    return subsets

trainsubset = get_subset(trainset,cscores['scores'])
testsubset = get_subset(testset,cscorestest)


if __name__ == "__main__":
    print(testsubset)
