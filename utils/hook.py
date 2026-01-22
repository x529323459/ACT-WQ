import torch
from torch.nn import Module
from mqbench.utils.hook import StopForwardException,DataSaverHook



def to_device(data, device='cpu'):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
        return data
    elif isinstance(data, list):
        for idx, _ in enumerate(data):
            data[idx] = to_device(data[idx], device)
        return data
    else:
        return data

def put_hook(modules,  store_inp=True, store_oup=True):
    for modulehook in modules:
        saver = DataSaverHook(store_input=store_inp, store_output=store_oup)
        handle = modulehook.module.register_forward_hook(saver)
        modulehook.sethook(saver)

def save_inp_oup_data(modules):
    inps =[]
    oups = []
    for modulehook in modules:
        modulehook.store_in_out()
        inps.append(modulehook.inp)
        oups.append(modulehook.oup)
    return inps,oups
