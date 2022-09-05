from asyncore import write
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
import torch
from torchvision import datasets, transforms
import logging
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
sys.path.append('./experiments-on-learning-speed')
from model import ResNet18
import random

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

tb_dir = './runs/differ-transforms-on-reverse-trainset/'
flag = sys.argv[1]
if flag =='raw':
    writer = SummaryWriter(f"{tb_dir}raw")
else:
    writer = SummaryWriter(f"{tb_dir}noraw")

seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
GPU = torch.cuda.is_available()
model = ResNet18(classnum=100)

if GPU:
    model.cuda()
    torch.backends.cudnn.benchmark = True
else:
    model.cpu()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.1,
    momentum=0.9,
    nesterov = True,
    weight_decay=5e-4,
)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[30,60,90],gamma=0.2,last_epoch=-1)

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
trainsetraw = datasets.CIFAR100(root='./CIFAR-100', train=True, download=True,)
testsetraw = datasets.CIFAR100(root='./CIFAR-100', train=False, download=True,)

def data_split(rawset,flag):
    dataset = []
    if flag=="train":
        for data in rawset:
            dataset.append((transform_train(data[0]),data[1]))
    else:
        for data in rawset:
            dataset.append((transform_test(data[0]),data[1]))
    return np.array(dataset,dtype=object)

trainset = data_split(trainsetraw,'train')
testset = data_split(testsetraw,'test')
trainset = trainset.tolist()
testset = testset.tolist()


trainsetraw = datasets.CIFAR100(root='./CIFAR-100', train=True, download=True, transform=transform_train)
testsetraw = datasets.CIFAR100(root='./CIFAR-100', train=False, download=True, transform=transform_test)

def train_each_epoch(epoch:int, optimizer,trainloader):
    model.train()

    print("*"*10)
    print(f"epoch{epoch+1}")
    train_loss = 0.0
    for data in trainloader:
        image = data[0]
        label = data[1]
        # input(image)
        if torch.cuda.is_available():
            image = Variable(image).cuda()
            label = Variable(label).cuda()
        else:
            image = Variable(image)
            label = Variable(label)
        
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output,label)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    logging.info(f"Finish {epoch+1} epoch,Loss {train_loss}")

@torch.no_grad()
def test(group,datasetsize,dataloader):
    acc = 0.0
    test_loss = 0.0

    model.eval()
    for i,data in enumerate(dataloader):
        image = data[0]
        label = data[1]
        writer.add_images(group,data[0],i )
        if GPU:
            image = Variable(image).cuda()
            label = Variable(label).cuda()
        else:
            image = Variable(image)
            label = Variable(label)
        output = model(image)
        loss = criterion(output,label)
        test_loss += loss.item()
        acc += (output.argmax(1).eq(label)).sum().item()
    acc_rate  = float(acc/datasetsize)
    logging.info(f"group = {group}, acc_rate={acc_rate},test_loss={test_loss} acc={acc}, datasetsize={datasetsize}")
    return acc_rate

if __name__ =='__main__':
    
    trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)
    testloaderraw = torch.utils.data.DataLoader(
    testsetraw, batch_size=128, shuffle=False, num_workers=2)
    trainloaderraw = torch.utils.data.DataLoader(
    trainsetraw, batch_size=128, shuffle=False, num_workers=2)
    
    # compare the two dataset:
    if flag =='raw':
        

        for epoch in range(100):
            train_each_epoch(epoch=epoch,optimizer=optimizer,trainloader=trainloaderraw)
            scheduler.step()
            print(optimizer.state_dict()['param_groups'][0]['lr'])
            writer.add_scalar('trainraw_acc',test(group='all',datasetsize = len(trainsetraw),dataloader=trainloaderraw),epoch)
            writer.add_scalar('testraw_acc',test(group='test',datasetsize = len(testsetraw),dataloader=testloaderraw),epoch)
        writer.close()
    else:
        
        
        for epoch in range(100):
            train_each_epoch(epoch=epoch,optimizer=optimizer,trainloader=trainloader)
            scheduler.step()
            
            print(optimizer.state_dict()['param_groups'][0]['lr'])
            writer.add_scalar('train_acc',test(group='all',datasetsize = len(trainset),dataloader=trainloader),epoch)
            writer.add_scalar('test_acc',test(group='test',datasetsize = len(testset),dataloader=testloader),epoch)
        writer.close()
    