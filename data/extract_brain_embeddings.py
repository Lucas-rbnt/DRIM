import pandas as pd

import torch
import csv
from drim.mri.transforms import tumor_transfo
from drim.mri.datasets import MRIProcessor
from drim.mri.models import MRIEncoder
from drim.utils import clean_state_dict
import tqdm
import os

# Load the data
data = pd.read_csv("./data/files/dataframe_brain.csv")
encoder = MRIEncoder(2, 512, False)
encoder.load_state_dict(
    clean_state_dict(torch.load("data/models/t1ce-flair_tumorTrue.pth")), strict=False
)
encoder.eval()
for mri_path in tqdm.tqdm(data.MRI):
    if pd.isna(mri_path):
        continue
    process = MRIProcessor(
        mri_path,
        tumor_centered=True,
        transform=tumor_transfo,
        modalities=["t1ce", "flair"],
        size=(64, 64, 64),
    )
    mri = process.process().unsqueeze(0)
    with torch.no_grad():
        embedding = encoder(mri)

    # Save the embedding
    with open(os.path.join(mri_path, "embedding.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow([str(i) for i in range(512)])
        writer.writerow(embedding.squeeze().numpy().tolist())
