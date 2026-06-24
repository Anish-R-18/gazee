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

    # column
    if x < WIDTH/3:
        col = 0
    elif x < 2*WIDTH/3:
        col = 1
    else:
        col = 2

    # row
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


# print("Loading fixation file...")
mat = loadmat(r"C:\Users\anish\OneDrive\Desktop\SRIP\gaze\predicting-human-gaze-beyond-pixels\data\eye\fixations.mat")

fixations = mat["fixations"]

results = []

for img_idx in tqdm(range(fixations.shape[0])):

    image_data = fixations[img_idx,0]

    image_name = image_data["img"][0,0][0]

    subjects = image_data["subjects"][0,0]

    region_duration = defaultdict(float)
    total_duration = 0

    for subj_idx in range(subjects.shape[0]):

        subj = subjects[subj_idx,0]

        xs = subj["fix_x"][0,0].flatten()
        ys = subj["fix_y"][0,0].flatten()
        ds = subj["fix_duration"][0,0].flatten()

        for x,y,d in zip(xs,ys,ds):

            region = get_region(x,y)

            region_duration[region] += float(d)
            total_duration += float(d)

    row = {
        "image": image_name
    }

    for region in REGIONS:

        if total_duration > 0:
            row[region] = region_duration[region] / total_duration
        else:
            row[region] = 0

    results.append(row)

df = pd.DataFrame(results)

df.to_csv(
    "image_dwell_proportions.csv",
    index=False
)

# print("\nSaved:")
# print("image_dwell_proportions.csv")

# print("\nExample:")
# print(df.head())
# df.mean(numeric_only=True).sort_values(ascending=False)
# print(df.describe())
df["sum"] = df[
    ["UL","UC","UR",
     "ML","MC","MR",
     "BL","BC","BR"]
].sum(axis=1)

print(df["sum"].describe())