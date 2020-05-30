#%%
import i2v
from PIL import Image
#%%

illust2vec = i2v.PytorchI2V(
            "illust2vec_tag_ver200.pth", "tag_list.json")
#%%

img = Image.open("images/miku.jpg")
x = illust2vec._estimate([img])
for i in x[0]:
    print(float(i))


# %%
