import torch.nn.functional as F
import torch.nn as nn
from analyse.measure import *

def mse_fuction(x,y): #计算x和y的mse
    return F.mse_loss(x, y)

def calculate_list_error(list1, list2, type):
    # 确保两个列表长度相同
    if len(list1) != len(list2):
        raise ValueError("两个列表的长度必须相同")
    if type == 'snr':
        func = torch_snr_error
    if type == 'mse':
        func = torch_mean_square_error
    # 计算MSE
    mse = 0
    count = 0
    for x, y in zip(list1, list2):
        if x == None or y == None:
            count += 1
            continue
        mse += func(x[0], y[0])
    return mse/len(list1)

def calculate_layer_mse(list1, list2):
    # 确保两个列表长度相同
    if len(list1) != len(list2):
        raise ValueError("两个列表的长度必须相同")
    # 计算MSE
    mse = []
    count = 0
    for x, y in zip(list1, list2):
        if x == None or y == None:
            count += 1
            continue
        result = mse_fuction(x[0], y[0])
        mse.append(result)
    return mse

def nonone_list(modules,fp_list): #因为有算子的hook是None，为了处理这种情况，多传了一个参数
    indices = []
    # 确保 fp_list 中的元素数量与 modules 中的元素数量相匹配
    if len(fp_list) != len(modules):
        raise ValueError("Length of fp_list must match length of modules")
        # 遍历 modules 和 fp_list
    for i, (module, fp) in enumerate(zip(modules, fp_list)):
        if fp is not None:
            indices.append(i)  # 添加当前索引到 conv_indices
    return indices


def split_list(modules,fp_list): #因为有算子的hook是None，为了处理这种情况，多传了一个参数
    # 初始化存储索引的列表
    conv_indices = []
    linear_indices = []
    # 确保 fp_list 中的元素数量与 modules 中的元素数量相匹配
    if len(fp_list) != len(modules):
        raise ValueError("Length of fp_list must match length of modules")
        # 遍历 modules 和 fp_list
    for i, (module, fp) in enumerate(zip(modules, fp_list)):
        if isinstance(module.module, nn.Conv2d):
            conv_indices.append(i)  # 添加当前索引到 conv_indices
        elif isinstance(module.module, nn.Linear) and fp is not None:
            linear_indices.append(i)  # 添加当前索引到 linear_indices（如果 fp 不是 None）
    return conv_indices, linear_indices

def total_loss_analyse(fp_modules,fp_list,q_list,type):
    mse = calculate_list_error(fp_list,q_list,type)
    print(f'total mse={mse}')
    fp_conv_list = []
    q_conv_list = []
    fp_linear_list = []
    q_linear_list = []
    index = 0
    for m in fp_modules:
        if isinstance(m.module, nn.Conv2d):
            fp_conv_list.append(fp_list[index])
            q_conv_list.append(q_list[index])
        elif isinstance(m.module, nn.Linear) and fp_list[index] != None:
            fp_linear_list.append(fp_list[index])
            q_linear_list.append(q_list[index])
        index += 1
    mse = calculate_list_error(fp_conv_list, q_conv_list,type)
    print(f'conv mse={mse}')
    mse = calculate_list_error(fp_linear_list, q_linear_list,type)
    print(f'linear mse={mse}')

def layer_analyse(index,fp_list,q_list):
    fps = [fp_list[i] for i in index]#取出属于某个算子的list
    qs = [q_list[i] for i in index]  # 取出属于某个算子的list
    error = []
    for x, y in zip(fps, qs):
        result = torch_mean_square_error(x[0], y[0])
        #result = torch_snr_error(x[0], y[0])
        error.append(result)
    return error

def calculate_average(list):
    sum = 0
    for x in list:
        sum+=x
    return sum/len(list)