#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"
# %%
import i2v
from PIL import Image
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
import pickle
import numpy as np
# %% データの用意
root = Path("/data/natsuki/training116/00030-v4-mirror-auto4-gamma100-batch64-noaug-resumecustom/011289")
assert root.is_dir()

BatchSize = 32
filelist = list(root.glob("*"))
filelist.sort()
#%% caffeのタグ使う版
illust2vec = i2v.make_i2v_with_chainer(
    "illust2vec_tag_ver200.caffemodel", "tag_list.json")
# %%
outs = list()
for i in tqdm(range((len(filelist)-1)//BatchSize+1)):
    imgs = [Image.open(fname) for fname in filelist[i*BatchSize:(i+1)*BatchSize]]
    outs += illust2vec.estimate_plausible_tags(imgs, threshold=0)
# %%
with open(root/".."/"tag.pkl", "wb") as f:
    pickle.dump(outs, f, protocol=4)
# %% len2のtupleからdictへの変換
for i in tqdm(range(len(outs))):
    for k in outs[i]:
        outs[i][k] = dict(outs[i][k])
# %% 4096の特徴量使う版
illust2vec = i2v.make_i2v_with_chainer("illust2vec_ver200.caffemodel")
# %%

outs = list()
for fname in tqdm(filelist): 
    imgs = [Image.open(fname)]
    result_real = illust2vec.extract_feature(imgs)
    outs.append(result_real)
# %%
outs = np.vstack(outs) # shape == (num, 4096)
# %%
np.save(root/".."/"tag.npy", outs)
outs2 = np.load(root/".."/"tag.npy")
assert np.all(outs == outs2)
# %% Pytorch使う版
illust2vec = i2v.PytorchI2V(
            "illust2vec_tag_ver200.pth", "tag_list.json")
img = Image.open("images/miku.jpg")
x = illust2vec._estimate([img])
print(x.shape)