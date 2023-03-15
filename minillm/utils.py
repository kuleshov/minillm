import torch.nn as nn
import urllib.request

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def download_file(url, path):
	print('Starting download')
	urllib.request.urlretrieve(url, path)
	print('Done')