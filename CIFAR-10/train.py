from unittest import result
import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import json

# 导入
from dataset import keys,trainloader,trainset,trainsdatasubsets,trainsdatasublaoders
from model import Net
import logging

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# 定义超参数
learning_rate = 1e-3
momentum = 0.9
num_epoches = 10
batch_size = 32

model = Net()
#判断是否有GPU, 如果使用GPU就将模型放到GPU中进行训练
use_gpu = torch.cuda.is_available()
if use_gpu:
    model.cuda()
print(model)


# 交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss()
# SGD梯度下降方法
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

def train(trainset , trainloader, batch_size = 32, epochs_num = 10, lr = 1e-3, momentum = 0.9):
    # 训练
    writer = SummaryWriter('runs/trains')
    train_acc_list = []
    for epoch in range(epochs_num):
        print("*"*10)
        print("epoch{}".format(epoch+1))
        train_loss = 0.0
        train_acc =0.0
        for i,data in enumerate(trainloader,0): # enumerate是python的内置函数，既获得索引也获得数据
        #for image, label in trainloader:
            image, label = data # data包含数据和标签信息
            print(image)
            # 将数据转换成Variable
            if use_gpu:
                image = Variable(image).cuda()
                label = Variable(label).cuda()
            else:
                image = Variable(image)
                label = Variable(label)

            # 要把梯度重新归零，因为反向传播过程中梯度会累加上一次循环的梯度
            optimizer.zero_grad()

            output = model(image)
            loss = criterion(output, label)
            train_loss += loss.item()*label.size(0) # 此处是什么意思？
            # train_loss += loss.item() 也可以
            _, pred = torch.max(output,1) # 返回每行的最大值的索引值，也就是预测出的可能性最大的类别的相应的标签
            train_correct = (pred == label).sum()
            train_acc += train_correct.item()

            loss.backward() # 反向传播
            optimizer.step() # 梯度更新
            
        # 每训练完一个周期打印一次平均损失和平均精度
        logging.info('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, train_loss / (len(trainset)), train_acc / (len(trainset))
        ))
        train_acc_list.append(train_acc)
        writer.add_scalar('train-loss',train_loss / (len(trainset)),epoch)
        writer.add_scalar('train_acc',train_acc / (len(trainset)), epoch)
    writer.close()
    return train_acc_list

if __name__ == '__main__':
    result = {}

    

    for i in range(len(keys)):
        result[keys[i]] = train(trainset=trainsdatasubsets[i]["datasubset"],trainloader=trainsdatasublaoders[i]['datasubloader'],epochs_num=200)
    result['all'] = train(trainset,trainloader,epochs_num = 200)

    
    with open('result.json','w') as f:
        json.dump(result,f)