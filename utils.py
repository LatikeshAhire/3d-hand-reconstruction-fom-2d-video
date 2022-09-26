import re

import numpy as np

import config
import copy


def compute_npx_error(prediction, gt, n):
    # computing n-px error
    gt = copy.deepcopy(gt)
    array = np.zeros(gt.shape)
    array[gt>0] = 1

    greater_than_one = len(array[array==1])

    disp_true = np.abs(gt-prediction)
    disp_true = disp_true*array
    error = len(disp_true[disp_true>3])

    return 1-float(error)/float(greater_than_one)


def readPFM(file):
    if not isinstance(file, str):
        file = file.numpy().decode()
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip().decode()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip().decode())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)

    data[data > (config.MAX_DISP - 1)] = config.MAX_DISP - 1
    return data


def mean_std(inputs):
    inputs = np.float32(inputs) / 255.
    inputs[:, :, 0] -= 0.485
    inputs[:, :, 0] /= 0.229
    inputs[:, :, 1] -= 0.456
    inputs[:, :, 1] /= 0.224
    inputs[:, :, 2] -= 0.406
    inputs[:, :, 2] /= 0.225
    return inputs
