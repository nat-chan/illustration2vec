# Illustration2Vec

``illustration2vec (i2v)`` is a simple library for estimating a set of tags and
extracting semantic feature vectors from given illustrations.
For details, please see
[our main paper](https://github.com/rezoo/illustration2vec/raw/master/papers/illustration2vec-main.pdf).

# Requirements

* Pre-trained models (``i2v`` uses Convolutional Neural Networks. Please download
  several pre-trained models from
  [here](https://github.com/rezoo/illustration2vec/releases),
  or execute ``get_models.sh`` in this repository).
* ``numpy`` and ``scipy``
* ``PIL`` (Python Imaging Library) or its alternatives (e.g., ``Pillow``) 
* ``skimage`` (Image processing library for python)

In addition to the above libraries and the pre-trained models, `i2v` requires
either ``caffe`` or ``chainer`` library. If you are not familiar with deep
learning libraries, we recommend to use ``chainer`` that can be installed
via ``pip`` command.

# How to use

In this section, we show two simple examples -- tag prediction and the the
feature vector extraction -- by using the following illustration [1].

![slide](images/miku.jpg)

[1] Hatsune Miku (初音ミク), © Crypton Future Media, INC.,
http://piapro.net/en_for_creators.html.
This image is licensed under the Creative Commons - Attribution-NonCommercial,
3.0 Unported (CC BY-NC).

## Tag prediction

``i2v`` estimates a number of semantic tags from given illustrations
in the following manner.
```python
import i2v
from PIL import Image

illust2vec = i2v.PytorchI2V(
    "illust2vec_tag_ver200.pth", "tag_list.json")

img = Image.open("images/miku.jpg")
illust2vec._estimate([img])
```

## Feature vector extraction
WIP

# License
The pre-trained models and the other files we have provided are licensed
under the MIT License.
