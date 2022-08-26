
from email.mime import image
from pyexpat import model
from unittest import result
from estimation_model import Inception
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import torch
import numpy as np
import numpy.random as npr

import jax.numpy as jnp
from jax import random

from tqdm import tqdm
import logging

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

GPU = torch.cuda.is_available()

def model_init(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = Inception(class_num=100)
    if GPU:
        model.cuda()
    else:
        model.cpu()
    return model

## the whole data test on the trained model
def correctness(model,dataset):
    correct = []
    batchsize = 1024
    with torch.no_grad():
        dataloader = DataLoader(dataset=dataset,batch_size=batchsize)
        for data in dataloader:
            image = data['image']
            label = data['label']
            if GPU:
                image = Variable(image).cuda()
                label = Variable(label).cuda()
            else:
                image = Variable(image)
                label = Variable(label)
            correct+=  (model(image).argmax(1) == label).cpu()
    return correct


## train the model by the subset_ratio dataset for num_epochs times
def subset_train(seed,subset_ratio,dataset,model):
    
    lr = 0.1
    momentum = 0.9
    num_epochs = 10
    batch_size = 1024

    num_total = len(dataset)
    num_train = int(num_total*subset_ratio)
    
    rng = npr.RandomState(seed)
    subset_idx = rng.choice(num_total,size = num_train,replace=False)
    trainset = np.array(dataset)[subset_idx]

    trainloader = DataLoader(trainset,batch_size=batch_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr = lr,
        momentum=momentum, 
    )
    for _ in range(num_epochs):
        for data in trainloader:
            image = data['image']
            label = data['label']
            if GPU:
                image = Variable(image).cuda()
                label = Variable(label).cuda()
            else:
                image = Variable(image)
                label = Variable(label)
            
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()

    trainset_correctness = correctness(model,dataset=dataset)
    trainset_mask = np.zeros(num_total,dtype=np.bool)
    trainset_mask[subset_idx] = True
    
    
    return trainset_mask, np.asarray(trainset_correctness)


## randomly choose diverse ratio dataset, 
# train the initial model on the ratio dataset,
# then test the trained model on the whole dataset 
 
def estimate_cscores(dataset):
    n_runs = 200
    subset_ratios = np.linspace(0.1,0.9,9)

    cscores = {}
    for ss_ratio in subset_ratios:
        results = []
        for i_run in tqdm(range(n_runs),desc=f"SS Ratio={ss_ratio:.2f}"):
            model = model_init(i_run)
            results.append(subset_train(i_run,ss_ratio,dataset,model))

        trainset_mask = np.vstack([ret[0] for ret in results])
        trainset_correctness = np.vstack([ret[1] for ret in results])
        inv_mask = np.logical_not(trainset_mask)

        cscores[ss_ratio] = (np.sum(trainset_correctness * inv_mask, axis=0) / 
                            np.maximum(np.sum(inv_mask, axis=0), 1e-10))
  
    return cscores
  
def data_split(rawset):
    dataset=[]

    for data in rawset:
        dataset.append({"image":data[0],"label":data[1]})
    return dataset

if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    trainset = datasets.CIFAR100(root='./CIFAR-100',train = True,download=True, transform=transform)
    testset = datasets.CIFAR100(root='./CIFAR-100',train=False,download=True,transform=transform)
    totalset = trainset+testset
    totalset = data_split(totalset)    
    
    npz_fn = 'cscores.npy'
    cscores = estimate_cscores(totalset)
    cscores = np.mean([x for x in cscores.values()], axis=0)
    np.save(npz_fn, cscores)

