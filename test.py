#%%
import i2v
from PIL import Image
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
import pickle
#%%
import i2v
from PIL import Image

illust2vec = i2v.make_i2v_with_chainer(
    "illust2vec_tag_ver200.caffemodel", "tag_list.json")
# %%
root = Path("/data/natsuki/getchu_s5")
assert root.is_dir()

BatchSize = 32
filelist = list(root.glob("*"))
outs = list()
for i in tqdm(range((len(filelist)-1)//BatchSize+1)):
    imgs = [Image.open(fname) for fname in filelist[i*BatchSize:(i+1)*BatchSize]]
    outs += illust2vec.estimate_plausible_tags(imgs, threshold=0)
# %%
for i in tqdm(range(len(outs))):
    for k in outs[i]:
        outs[i][k] = dict(outs[i][k])
# %%
with open((root/f"/data/natsuki/getchu_s5_i2v.pkl"), "wb") as f:
    pickle.dump(dict(zip(list(map(str, filelist)), outs)), f, protocol=4)
# %%
illust2vec = i2v.PytorchI2V(
            "illust2vec_tag_ver200.pth", "tag_list.json")
img = Image.open("images/miku.jpg")
x = illust2vec._estimate([img])
print(x.shape)
# %%
# %%
