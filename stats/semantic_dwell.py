import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

ATTR_NAMES = [
    "text",
    "face",
    "emotion",
    "sound",
    "smell",
    "taste",
    "touch",
    "motion",
    "operability",
    "watchability",
    "touched",
    "gazed"
]

print("Loading fixation data...")
fix_mat = loadmat(r"C:\Users\anish\OneDrive\Desktop\SRIP\gaze\predicting-human-gaze-beyond-pixels\data\eye\fixations.mat")
fixations = fix_mat["fixations"]

print("Loading attribute data...")
f = h5py.File(r"C:\Users\anish\OneDrive\Desktop\SRIP\gaze\predicting-human-gaze-beyond-pixels\data\attrs.mat", "r")

attrs = f["attrs"]

results = []

for img_idx in tqdm(range(700)):

    # ---------- image ----------
    img_ref = attrs[0, img_idx]
    img_group = f[img_ref]

    image_name = ''.join(
        chr(c[0])
        for c in img_group["img"][:]
    )

    # ---------- objects ----------
    objs = img_group["objs"]

    object_masks = []
    object_features = []

    for j in range(objs.shape[1]):

        obj_ref = objs[0, j]
        obj = f[obj_ref]

        mask = obj["map"][:].astype(bool)

        feat = obj["features"][:].flatten()

        object_masks.append(mask)
        object_features.append(feat)

    # ---------- fixation data ----------
    eye_img = fixations[img_idx, 0]

    subjects = eye_img["subjects"][0,0]

    attr_duration = np.zeros(12)

    total_duration = 0

    for s in range(subjects.shape[0]):

        subj = subjects[s,0]

        xs = subj["fix_x"][0,0].flatten()
        ys = subj["fix_y"][0,0].flatten()
        ds = subj["fix_duration"][0,0].flatten()

        for x,y,d in zip(xs,ys,ds):

            x = int(round(x))
            y = int(round(y))

            if x < 0 or x >= 800:
                continue

            if y < 0 or y >= 600:
                continue

            total_duration += float(d)

            # check every object
            for mask,feat in zip(
                object_masks,
                object_features
            ):

                if mask[x,y]:

                    attr_duration += feat * float(d)

    row = {
        "image": image_name
    }

    if total_duration > 0:

        for k,name in enumerate(ATTR_NAMES):

            row[name] = (
                attr_duration[k] /
                total_duration
            )

    else:

        for name in ATTR_NAMES:
            row[name] = 0

    results.append(row)

df = pd.DataFrame(results)

df.to_csv(
    "semantic_dwell.csv",
    index=False
)

print(df.head())
print(df.describe())