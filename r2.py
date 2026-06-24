# from scipy.io import loadmat

# from scipy.io import loadmat

# d = loadmat(r"C:\Users\anish\OneDrive\Desktop\SRIP\gaze\predicting-human-gaze-beyond-pixels\data\mouse_amt\1001.mat")

# f = d["fixations"]

# print(type(f))
# print(f.shape)
# print(f.dtype)

# print(f[0,0])

# d = loadmat(r"C:\Users\anish\OneDrive\Desktop\SRIP\gaze\predicting-human-gaze-beyond-pixels\data\mouse_lab\1001.mat")
# f = d["fixations"]

# print(type(f))
# print(f.shape)
# print(f.dtype)

# print(f[0,0])

# import h5py

# f = h5py.File(r"C:\Users\anish\OneDrive\Desktop\SRIP\gaze\predicting-human-gaze-beyond-pixels\data\attrs.mat", "r")

# print(f["attrNames"])
# print(f["attrs"])

# refs = f["attrNames"]

# for i in range(refs.shape[0]):
#     ref = refs[i,0]
#     print(i, f[ref][()])

# for i in range(refs.shape[0]):
#     ref = refs[i,0]
#     name = ''.join(chr(c[0]) for c in f[ref][:])
#     print(i, name)

# print(list(f.keys()))

# print(d["attrNames"].shape)
# print(d["attrs"].shape)

# print(d["attrNames"][:10])

# print(type(d["attrs"]))
# print(d["attrs"][:5])
# for key in f.keys():
#     print(key, f[key])
# print(list(f.keys()))


import h5py

f = h5py.File(r"C:\Users\anish\OneDrive\Desktop\SRIP\gaze\predicting-human-gaze-beyond-pixels\data\attrs.mat","r")

attrs = f["attrs"]

ref = attrs[0,0]

# print(type(ref))
# print(ref)

obj = f[ref]

print(obj)

if isinstance(obj, h5py.Group):
    print("GROUP")
    print(list(obj.keys()))

elif isinstance(obj, h5py.Dataset):
    print("DATASET")
    print(obj.shape)
    print(obj[:10])

# print("img =", obj["img"])
# print("objs =", obj["objs"])

# print(type(obj["objs"]))
# print(obj["objs"].shape)

objs_ref = obj["objs"][0,0]

# print(objs_ref)

o = f[objs_ref]

# print(o)

img_ref = obj["img"][0,0]

# print(img_ref)

img_obj = f[img_ref]

# print(img_obj)

img_ds = obj["img"]

name = ''.join(chr(c[0]) for c in img_ds[:])

print(name)