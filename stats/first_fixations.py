from scipy.io import loadmat
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

WIDTH = 800
HEIGHT = 600

REGIONS = [
    "UL","UC","UR",
    "ML","MC","MR",
    "BL","BC","BR"
]

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


mat = loadmat(r"C:\Users\anish\OneDrive\Desktop\SRIP\gaze\predicting-human-gaze-beyond-pixels\data\eye\fixations.mat")
fixations = mat["fixations"]

results = []

for img_idx in tqdm(range(fixations.shape[0])):

    image_data = fixations[img_idx,0]

    image_name = image_data["img"][0,0][0]

    subjects = image_data["subjects"][0,0]

    counts = defaultdict(int)

    n_subjects = subjects.shape[0]

    for subj_idx in range(n_subjects):

        subj = subjects[subj_idx,0]

        xs = subj["fix_x"][0,0].flatten()
        ys = subj["fix_y"][0,0].flatten()

        n_fix = len(xs)

        for fixno in [1,2,3]:

            if n_fix >= fixno:

                x = xs[fixno-1]
                y = ys[fixno-1]

                region = get_region(x,y)

                counts[f"Fix{fixno}_{region}"] += 1

    row = {"image": image_name}

    for fixno in [1,2,3]:
        for region in REGIONS:

            key = f"Fix{fixno}_{region}"

            row[key] = counts[key] / n_subjects

    results.append(row)

df = pd.DataFrame(results)

df.to_csv(
    "first_fixation_regions.csv",
    index=False
)

print(df.head())