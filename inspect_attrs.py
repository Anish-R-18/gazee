import h5py

f = h5py.File(r"C:\Users\anish\OneDrive\Desktop\SRIP\gaze\predicting-human-gaze-beyond-pixels\data\attrs.mat", "r")

# print("="*60)
# print("TOP LEVEL KEYS")
# print("="*60)
# print(list(f.keys()))

# print("\n" + "="*60)
# print("ATTRIBUTE NAMES")
# print("="*60)

# attrNames = f["attrNames"]

# for i in range(attrNames.shape[0]):
#     ref = attrNames[i,0]
#     name = ''.join(chr(c[0]) for c in f[ref][:])
#     print(i, name)

# print("\n" + "="*60)
# print("FIRST IMAGE")
# print("="*60)

# attrs = f["attrs"]

# img_ref = attrs[0,0]
# img_group = f[img_ref]

# print("Image group keys:")
# print(list(img_group.keys()))

# print("\nImage name:")

# img_name = ''.join(chr(c[0]) for c in img_group["img"][:])
# print(img_name)

# print("\nObjects dataset:")
# print(img_group["objs"])
# print("shape =", img_group["objs"].shape)

# objs = img_group["objs"]

# print("\n" + "="*60)
# print("FIRST OBJECT")
# print("="*60)

# obj_ref = objs[0,0]

# obj = f[obj_ref]

# print("Object keys:")
# print(list(obj.keys()))

# for k in obj.keys():

#     print("\n-----", k, "-----")

#     try:
#         item = obj[k]

#         print("type:", type(item))
#         print("shape:", item.shape)

#         try:
#             print(item[:10])
#         except:
#             pass

#     except Exception as e:
#         print("ERROR:", e)
import numpy as np
attrs = f["attrs"]

img_ref = attrs[0,0]
img_group = f[img_ref]

objs = img_group["objs"]

for i in range(objs.shape[1]):

    obj_ref = objs[0,i]

    obj = f[obj_ref]

    feat = obj["features"][:].flatten()

    print("Object", i)

    active = np.where(feat > 0)[0]

    print(active)