from torch import nn 
import torch
from plnn.modules import View, Flatten
from torch.nn.parameter import Parameter
from plnn.model import simplify_network
import random
import copy
import json

'''
This file contains all model structures we have considered
'''

## original kw small model
## 14x14x16 (3136) --> 7x7x32 (1568) --> 100 --> 10 ----(4804 ReLUs)
def mnist_model(): 
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

## 14*14*8 (1568) --> 14*14*8 (1568) --> 14*14*8 (1568) --> 392 --> 100 (5196 ReLUs)
def mnist_model_deep():
    model = nn.Sequential(
        nn.Conv2d(1, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

# first medium size model 14x14x4 (784) --> 7x7x8 (392) --> 50 --> 10 ----(1226 ReLUs)
# robust error 0.068
def mnist_model_m1():
    model = nn.Sequential(
        nn.Conv2d(1, 4, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*7*7,50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    return model


# increase the mini model by increasing the number of channels
## 8x8x8 (512) --> 4x4x16 (256) --> 50 (50) --> 10 (818)
def mini_mnist_model_m1():
    model = nn.Sequential(
        nn.Conv2d(1, 8, 2, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 2, stride=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(4*4*16,50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    return model


# without the extra 50-10 layer (originally, directly 128-10, robust error is around 0.221)
## 8x8x4 (256) --> 4x4x8 (128) --> 50 --> 10 ---- (434 ReLUs)
def mini_mnist_model(): 
    model = nn.Sequential(
        nn.Conv2d(1, 4, 2, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 8, 2, stride=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*4*4,50),
        nn.ReLU(),
        nn.Linear(50,10),
    )
    return model

#### CIFAR

# 32*32*32 (32768) --> 32*16*16 (8192) --> 64*16*16 (16384) --> 64*8*8 (4096) --> 512 --> 512 
# 54272 ReLUs
def large_cifar_model(): 
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*8*8,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    return model

# 16*16*16 (4096) --> 32*8*8 (2048) --> 100 
# 6244 ReLUs
# wide model
def cifar_model():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

# 16*16*8 (2048) -->  16*16*8 (2048) --> 16*16*8 (2048) --> 512 --> 100
# 6756 ReLUs
#deep model
def cifar_model_deep():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*8*8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

# 16*16*8 (2048) --> 16*8*8 (1024) --> 100 
# 3172 ReLUs (small model)
def cifar_model_m2():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(16*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

# 16*16*4 (1024) --> 8*8*8 (512) --> 100 
def cifar_model_m1(): 
    model = nn.Sequential(
        nn.Conv2d(3, 4, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*8*8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model






def add_single_prop(layers, gt, cls):
    '''
    gt: ground truth lable
    cls: class we want to verify against
    '''
    additional_lin_layer = nn.Linear(10, 1, bias=True)
    lin_weights = additional_lin_layer.weight.data
    lin_weights.fill_(0)
    lin_bias = additional_lin_layer.bias.data
    lin_bias.fill_(0)
    lin_weights[0, cls] = -1
    lin_weights[0, gt] = 1

    #verif_layers2 = flatten_layers(verif_layers1,[1,14,14])
    final_layers = [layers[-1], additional_lin_layer]
    final_layer  = simplify_network(final_layers)
    verif_layers = layers[:-1] + final_layer
    for layer in verif_layers:
        for p in layer.parameters():
            p.requires_grad = False

    return verif_layers

    


def load_cifar_1to1_exp(model, idx, test = None, cifar_test = None):
    if model=='cifar_base_kw':
        model_name = './models/cifar_base_kw.pth'
        model = cifar_model_m2()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='cifar_wide_kw':
        model_name = './models/cifar_wide_kw.pth'
        model = cifar_model()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='cifar_deep_kw':
        model_name = './models/cifar_deep_kw.pth'
        model = cifar_model_deep()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    else:
        raise NotImplementedError

    if cifar_test is None:
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.225, 0.225, 0.225])
        cifar_test = datasets.CIFAR10('./cifardata/', train=False,transform=transforms.Compose([transforms.ToTensor(), normalize]))

    x,y = cifar_test[idx]
    x = x.unsqueeze(0)
    # first check the model is correct at the input
    y_pred = torch.max(model(x)[0], 0)[1].item()
    print('predicted label ', y_pred, ' correct label ', y)
    if  y_pred != y: 
        print('model prediction is incorrect for the given model')
        return None, None, None
    else: 
        if test ==None:
            choices = list(range(10))
            choices.remove(y_pred)
            test = random.choice(choices)

        print('tested against ',test)
        for p in model.parameters():
            p.requires_grad =False

        layers = list(model.children())
        added_prop_layers = add_single_prop(layers, y_pred, test)
        return x, added_prop_layers, test





