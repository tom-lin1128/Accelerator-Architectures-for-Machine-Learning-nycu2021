import numpy as np
from PIL import Image
import scipy
from numba import jit
import torch
import torch.nn as nn


Layers = ['Conv1', 'ReLU', 'Conv2', 'ReLU', 'Pooling', 
          'Conv3', 'ReLU', 'Conv4', 'ReLU', 'Pooling',
          'Conv5', 'ReLU', 'Conv6', 'ReLU', 'Conv7',
          'ReLU', 'Pooling', 'Conv8', 'ReLU', 'Conv9',
          'ReLU', 'Conv10', 'ReLU', 'Pooling', 'Conv11',
          'ReLU', 'Conv12', 'ReLU', 'Conv13', 'ReLU',
          'Pooling', 'AdaptivePooling', 'Fc14', 'ReLU',
          'Fc15', 'ReLU', 'Fc16']

im = Image.open('input.jpg')
image = np.asarray(im, dtype='f4')
image = image/255.0

image = np.swapaxes(image, 1, 2)
image = np.swapaxes(image, 0, 1)

def Padding(input_data):
    channel_n ,H , W = input_data.shape[0], input_data.shape[1], input_data.shape[2]
    zeros = [0]*(W+2)
    dataPad = input_data.tolist()
    
    for ch in range(channel_n):
        for i in range(H):
            dataPad[ch][i].insert(0, 0)
            dataPad[ch][i].append(0)
        dataPad[ch].insert(0, zeros)
        dataPad[ch].append(zeros)
    
    return np.asarray(dataPad)

@jit
def Convolution(input_data, weight, bias):
    H, W = input_data.shape[1], input_data.shape[2]
    filter_n, channel_n, kernel_h, kernel_w = weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]

    #convolution
    output = []
    for fn in range(filter_n):
        single_map = []
        for i in range(H-kernel_h+1):
            row = []
            for j in range(W-kernel_w+1):
                temp = 0.0
                for ch in range(channel_n):
                    for k in range(kernel_h):
                        for l in range(kernel_w):
                            temp += input_data[ch][i+k][j+l] * weight[fn][ch][k][l]
                row.append(temp+bias[fn])
            single_map.append(row)
        output.append(single_map)
    return np.asarray(output)

def  Conv2D(input_data, weight, bias):
    dataPad = Padding(input_data)
    return Convolution(dataPad, weight, bias)

def Pooling(input_data):
    channel_n, H, W = input_data.shape[0], input_data.shape[1], input_data.shape[2]
    K, L = 2, 2
    HK = H // K
    WL = W // L
    output = []
    if H % 2 == 0 and W % 2 == 0:
        for i in range(channel_n):
            mat = input_data[i]
            soln = mat[:HK*K, :WL*L].reshape(HK, K, WL, L).max(axis=(1, 3))
            output.append(soln)
    else:
        for i in range(channel_n):
            mat = input_data[i]
            Q1 = mat[:HK * K, :WL * L].reshape(HK, K, WL, L).max(axis=(1,3))
            Q2 = mat[HK * K:, :WL * L].reshape(-1, WL, L).max(axis=2)
            Q3 = mat[:HK * K, WL * L:].reshape(HK, K, -1).max(axis=1)
            Q4 = mat[HK * K:, WL * L:].max()
            soln = np.vstack([np.c_[Q1, Q3], np.c_[Q2, Q4]])
            soln = mat[:HK*K, :WL*L].reshape(HK, K, WL, L).max(axis=(1, 3))
            output.append(soln)
    return np.asarray(output)


@jit
def Fc(input_data, weight, bias):
    data = input_data.flatten()
    output = []
    for i in range(weight.shape[0]):
        temp = 0.0
        for j in range(weight.shape[1]):
            temp += data[j] * weight[i][j]
        output.append(temp)
    return np.asarray(output) + bias


def ReLU(input_data):
    if(len(input_data.shape)==3):
        channel_n, H, W = input_data.shape[0], input_data.shape[1], input_data.shape[2]
        output = []
        for i in range(channel_n):
            single_map = []
            for j in range(H):
                row = []
                for k in range(W):
                    row.append(input_data[i][j][k] if input_data[i][j][k] >= 0 else 0)
                single_map.append(row)
            output.append(single_map)
    else:
        output = []
        for i in range(input_data.shape[0]):
            output.append(input_data[i] if input_data[i] >= 0 else 0)
    return np.asarray(output)

def FCReLU(input_data):
    output = []
    print('shape:', input_data.shape)
    for i in range(input_data.shape[0]):
        output.append(input_data[i] if input_data[i] >= 0 else 0)
    return np.asarray(output)

def AdaptiveAvgPool2D(input_data):
    data = torch.from_numpy(input_data)
    m = nn.AdaptiveAvgPool2d(output_size=(7, 7))
    input = data
    output = m(input)
    return np.asarray(output)

data = image
for layer in Layers:
    print(layer)
    if layer.find('Conv') != -1:
        weights_file = layer + '_weights.npy'
        bias_file = layer + '_bias.npy'
        weight = np.load(weights_file)
        bias = np.load(bias_file)
        data = Conv2D(data, weight, bias)
        #data = ReLU(data)
    elif layer.find('Fc') != -1:
        weights_file = layer + '_weights.npy'
        bias_file = layer + '_bias.npy'
        weight = np.load(weights_file)
        bias = np.load(bias_file)
        data = Fc(data, weight, bias)

    elif layer == 'Pooling':
        data = Pooling(data)
    elif layer == 'ReLU':
        data = ReLU(data)
    else:
        data = AdaptiveAvgPool2D(data)

print(data)
print(data[452], data.max())

