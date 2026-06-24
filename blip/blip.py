from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
import pandas as pd
from tqdm import tqdm

device = "cuda"

print("Loading BLIP model...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

image_folder = r"C:\Users\anish\OneDrive\Desktop\SRIP\gaze\predicting-human-gaze-beyond-pixels\data\stimuli"      # change this
output_csv = "captions.csv"

results = []

files = [
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Found {len(files)} images")

for file in tqdm(files):

    path = os.path.join(image_folder, file)

    try:
        image = Image.open(path).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt"
        ).to(device)

        output = model.generate(
            **inputs,
            max_new_tokens=40
        )

        caption = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        results.append({
            "image": file,
            "caption": caption
        })

    except Exception as e:
        print(f"Error: {file} -> {e}")

df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)

print(f"Saved {len(results)} captions to {output_csv}")