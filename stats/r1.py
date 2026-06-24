import h5py
import numpy as np

f = h5py.File(r"C:\Users\anish\OneDrive\Desktop\SRIP\gaze\predicting-human-gaze-beyond-pixels\data\attrs.mat", "r")

attrs = f["attrs"]

img_group = f[attrs[0,0]]

obj_ref = img_group["objs"][0,0]

obj = f[obj_ref]

mask = obj["map"][:]

print(mask.shape)

ys, xs = np.where(mask == 1)

print("x range:", xs.min(), xs.max())
print("y range:", ys.min(), ys.max())
print("pixels:", len(xs))