from scipy.io import loadmat
import pandas as pd
import numpy as np
from tqdm import tqdm

WIDTH = 800
HEIGHT = 600

def get_region(x, y):

    if x < WIDTH/3:
        col = 0
    elif x < 2*WIDTH/3:
        col = 1
    else:
        col = 2

    if y < HEIGHT/3:
        row = 0
    elif y < 2*HEIGHT/3:
        row = 1
    else:
        row = 2

    grid = [
        ["UL","UC","UR"],
        ["ML","MC","MR"],
        ["BL","BC","BR"]
    ]

    return grid[row][col]


def shannon_entropy(sequence):

    if len(sequence) == 0:
        return 0

    unique, counts = np.unique(sequence, return_counts=True)

    probs = counts / counts.sum()

    return -(probs * np.log2(probs)).sum()


mat = loadmat(r"C:\Users\anish\OneDrive\Desktop\SRIP\gaze\predicting-human-gaze-beyond-pixels\data\eye\fixations.mat")
fixations = mat["fixations"]

results = []

for img_idx in tqdm(range(fixations.shape[0])):

    img = fixations[img_idx,0]

    image_name = img["img"][0,0][0]

    subjects = img["subjects"][0,0]

    entropies = []

    for s in range(subjects.shape[0]):

        subj = subjects[s,0]

        xs = subj["fix_x"][0,0].flatten()
        ys = subj["fix_y"][0,0].flatten()

        regions = [
            get_region(x,y)
            for x,y in zip(xs,ys)
        ]

        H = shannon_entropy(regions)

        entropies.append(H)

    results.append({
        "image": image_name,
        "mean_entropy": np.mean(entropies),
        "std_entropy": np.std(entropies),
        "min_entropy": np.min(entropies),
        "max_entropy": np.max(entropies)
    })

df = pd.DataFrame(results)

df.to_csv(
    "fixation_entropy.csv",
    index=False
)

# print(df.head())

# print("\nSummary:")
# print(df[["mean_entropy"]].describe())

# print(df.sort_values("mean_entropy").head(10))
print(
    df.sort_values(
        "mean_entropy",
        ascending=False
    ).head(10)
)