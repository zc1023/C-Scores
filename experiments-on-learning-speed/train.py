
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter
import json

# 导入
from dataset import trainsubset,trainset,testsubset,testset
from model import Net,ResNet18,ResNet34,ResNetEasy
import logging
import os
import sys
import numpy as np

#init
seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
GPU = torch.cuda.is_available()
model = ResNet18(classnum=100)
criterion = torch.nn.CrossEntropyLoss()
if GPU:
    model.cuda()
    torch.backends.cudnn.benchmark = True
else:
    model.cpu()

flag = sys.argv[1]
if flag == 'sgd':
    lr = float(sys.argv[2])
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr = lr,
        momentum=0.9,
        nesterov = True,
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[60,120,160],gamma=0.2,last_epoch=-1)
else:
    lr = float(flag)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = lr,
    )


def train_each_epoch(epoch:int, optimizer,trainloader):
    model.train()

    print("*"*10)
    print(f"epoch{epoch+1}")
    train_loss = 0.0
    for data in trainloader:
        image = data[0]
        label = data[1]
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
    for data in dataloader:
        image = data[0]
        label = data[1]
        # print(image)
        # print(label)
        if GPU:
            image = Variable(image).cuda()
            label = Variable(label).cuda()
        else:
            image = Variable(image)
            label = Variable(label)
        output = model(image)
        loss = criterion(output,label)
        test_loss += loss.item()
        acc += (output.argmax(1) == label).sum()
    acc_rate  = float(acc/datasetsize)
    # print(type(acc_rate))
    logging.info(f"group = {group}, acc_rate={acc_rate},test_loss={test_loss} acc={acc}, datasetsize={datasetsize}")
    return acc_rate

def init(checkpointfile):
    if os.path.exists(checkpointfile):
        checkpoint = torch.load(checkpointfile)
    
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        result = checkpoint['result']
        testresult = checkpoint['testresult']
        epoch_have_done = checkpoint['epoch']

    else:
        result = {}
        result["all"] = []
        result['test'] = []
        testresult = {}
        testresult["all"] = []
        testresult['test'] = []
        subset_ratios = np.linspace(0, 1, 21)
        for i in range(len(subset_ratios)-1):
            result[f"{round(subset_ratios[i],4)}--{round(subset_ratios[i+1],4)}"]=[]
            testresult[f"{round(subset_ratios[i],4)}--{round(subset_ratios[i+1],4)}"]=[]
        epoch_have_done = -1
    return result,testresult,epoch_have_done



if __name__ == "__main__":
    batchsize = 128


    # test(group="all",dataset=trainset,dataloader=trainloader)
    if flag == 'sgd':
        jsonfile = f'CIFAR-100_no_split_no_shuffle_transform_horizontal_differ_between_train_and_test_ResNet18_SGD_{lr}_{batchsize}'
    else:
        jsonfile = f'CIFAR-100_ResNet18_Adam_{lr}_{batchsize}'
    checkpointfile = './checkpoint/checkpoint'+jsonfile+'.tar'
    
    # print(model.state_dict())
    result,testresult,epoch_have_done = init(checkpointfile)
    # print(model.state_dict())

    trainloader = DataLoader(trainset,batch_size=batchsize,shuffle=True, num_workers=2)
    testloader = DataLoader(testset,batch_size=batchsize,shuffle=False, num_workers=2)
    for subset in trainsubset:
        subset['dataloader'] = DataLoader(subset['dataset'],batch_size=batchsize,shuffle=False, num_workers=2)
    for subset in testsubset:
        subset['dataloader'] = DataLoader(subset['dataset'],batch_size=batchsize,shuffle=False, num_workers=2)

    
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr = lr,
    #     momentum = 0.9,
    #     nesterov = True,
    #     weight_decay=5e-4,
    # )

    for epoch in range(epoch_have_done+1,200):
        
        train_each_epoch(epoch=epoch,optimizer=optimizer,trainloader=trainloader)
        result['all'].append(test(group="all",datasetsize=len(trainset),dataloader=trainloader))
        testresult['all'].append(result['all'][-1])
        for subset in trainsubset:
            result[subset['key']].append(test(group=subset['key'],datasetsize=len(subset['dataset']),dataloader=subset['dataloader']))
        result['test'].append(test(group='test',datasetsize=len(testset),dataloader=testloader))
        testresult['test'].append(result['test'][-1])
        for subset in testsubset:
            testresult[subset['key']].append(test(group=subset['key'],datasetsize=len(subset['dataset']),dataloader=subset['dataloader']))
        
        if flag == 'sgd':
            scheduler.step()
        
        
        torch.save({'epoch':epoch,
            'result':result,
            'testresult':testresult,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            },
            checkpointfile)
    for res in result:
        if len(res) == 0:
            result.remove(res)
    for res in testresult:
        if len(res) == 0:
            testresult.remove(res)
    
    if not os.path.exists("./data/"):
        os.mkdir("./data")
    if not os.path.exists("./test/"):
        os.mkdir("./test")
    with open (f"./data/{jsonfile}.json",'w')as f:
        json.dump(result,f)
    with open (f"./test/{jsonfile}.json",'w')as f:
        json.dump(testresult,f)
        

    
