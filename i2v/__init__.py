from i2v.base import Illustration2VecBase

caffe_available = False
chainer_available = False
pytorch_available = False

try:
    from i2v.caffe_i2v import CaffeI2V, make_i2v_with_caffe
    caffe_available = True
except ImportError:
    pass

try:
    from i2v.chainer_i2v import ChainerI2V, make_i2v_with_chainer
    chainer_available = True
except ImportError:
    pass

try:
    from i2v.pytorch_i2v import PytorchI2V, make_i2v_with_pytorch
    pytorch_available = True
except ImportError:
    pass

if not any([caffe_available, chainer_available, pytorch_available]):
    raise ImportError('i2v requires caffe, chainer or pytorch package')
