from i2v.base import Illustration2VecBase
import json
import warnings
import numpy as np
from scipy.ndimage import zoom
from skimage.transform import resize

import chainer
from chainer import Variable
from chainer.functions import average_pooling_2d, sigmoid
from chainer.links.caffe import CaffeFunction

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class PytorchI2V(nn.Module):
    def __init__(self, param_path, tag_path):
        super().__init__()
        if tag_path is not None:
            tags = json.loads(open(tag_path, 'r').read())
            assert(len(tags) == 1539)
            self.tags = np.array(tags)
            self.index = {t: i for i, t in enumerate(tags)}
        inplace = False
        self.mean = np.array([ 164.76139251,  167.47864617,  181.13838569])
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU(inplace)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU(inplace)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU(inplace)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv6_1 = nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu6_1 = nn.ReLU(inplace)
        self.conv6_2 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu6_2 = nn.ReLU(inplace)
        self.conv6_3 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu6_3 = nn.ReLU(inplace)
        self.conv6_4 = nn.Conv2d(1024, 1539, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool6 = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.prob = nn.Sigmoid()
        self.load_state_dict(torch.load(param_path))


    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.pool4(x)
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.pool5(x)
        x = self.conv6_1(x)
        x = self.relu6_1(x)
        x = self.conv6_2(x)
        x = self.relu6_2(x)
        x = self.conv6_3(x)
        x = self.relu6_3(x)
        x = self.conv6_4(x)
        x = self.pool6(x)
        x = self.prob(x)
        return x

    def _forward(self, inputs):
        shape = [len(inputs), 224, 224, 3]
        input_ = np.zeros(shape, dtype=np.float32)
        for ix, in_ in enumerate(inputs):
            input_[ix] = self.resize_image(in_, shape[1:])
        input_ = input_[:, :, :, ::-1]  # RGB to BGR
        input_ -= self.mean  # subtract mean
        input_ = input_.transpose((0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)
        x = torch.from_numpy(input_.copy())
        return self.forward(x)

    def resize_image(self, im, new_dims, interp_order=1):
        # NOTE: we import the following codes from caffe.io.resize_image()
        if im.shape[-1] == 1 or im.shape[-1] == 3:
            im_min, im_max = im.min(), im.max()
            if im_max > im_min:
                # skimage is fast but only understands {1,3} channel images
                # in [0, 1].
                im_std = (im - im_min) / (im_max - im_min)
                resized_std = resize(im_std, new_dims, order=interp_order, mode='constant')
                resized_im = resized_std * (im_max - im_min) + im_min
            else:
                # the image is a constant -- avoid divide by 0
                ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]),
                               dtype=np.float32)
                ret.fill(im_min)
                return ret
        else:
            # ndimage interpolates anything but more slowly.
            scale = tuple(np.array(new_dims) / np.array(im.shape[:2]))
            resized_im = zoom(im, scale + (1,), order=interp_order)
        return resized_im.astype(np.float32)

    def _convert_image(self, image):
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 2:
            # convert a monochrome image to a color one
            ret = np.empty((arr.shape[0], arr.shape[1], 3), dtype=np.float32)
            ret[:] = arr.reshape(arr.shape[0], arr.shape[1], 1)
            return ret
        elif arr.ndim == 3:
            # if arr contains alpha channel, remove it
            return arr[:,:,:3]
        else:
            raise TypeError('unsupported image specified')

    def _extract(self, inputs, layername):
        if layername == 'prob':
            return self._forward(inputs)
        else:
            raise Exception

    def _estimate(self, images):
        assert(self.tags is not None)
        imgs = [self._convert_image(img) for img in images]
        prob = self._extract(imgs, layername='prob')
        prob = prob.reshape(prob.shape[0], -1)
        return prob
