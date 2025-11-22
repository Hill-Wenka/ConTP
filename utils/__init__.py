import os
import sys

from omegaconf import OmegaConf
from .file import read_yaml

'''
Set the PYTHON_PATH and store constants
'''
# project root path: '/home/hew/python/ConTP/'
utils_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(utils_path)[0]

if root_path not in sys.path:
    sys.path.append(root_path)
    sys.path.append(utils_path)
    print('============================= add root_path to sys.path =============================')
    print('root_path:', root_path)
    print('======================================================================================')

paths = OmegaConf.create()
paths['root'] = os.path.join(root_path, '')
paths['data'] = os.path.join(root_path, 'data', '')
paths['dataset'] = os.path.join(root_path, 'dataset', '')
paths['script'] = os.path.join(root_path, 'script', '')
paths['temp'] = os.path.join(root_path, 'temp', '')
paths['utils'] = os.path.join(utils_path, '')

env = read_yaml(os.path.join(utils_path, 'config.yaml'))
