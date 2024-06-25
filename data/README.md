# Data
All data used in this study are from The Cancer Genome Atlas (TCGA) program [(Weinstein et al., Nat Genet 2013)](https://www.nature.com/articles/ng.2764). They are all publicly available on the [GDC Data Portal](https://portal.gdc.cancer.gov/).

# Structure
The structure of this given part is organised as follows
```
├── scripts
│   ├── gdc_manifest_20230619_DNAm_GBM_LGG.txt
│   ├── gdc_manifest_20230918_WSI_LGG.txt
│   ├── gdc_manifest_20231124_WSI_GBM.txt
│   ├── gdc_manifest_20231203_RNA_GBM_LGG.txt
├── files
│   ├── clinical_data.tsv
│   ├── supplementary_data.tsv
│   ├── ... <- preprocessed dataframes created from preprocessing.ipynb
├── mappings
│   ├── wsi_mapping.json
│   ├── BraTS2023_2017_GLI_Mapping.xlsx
├── models
│   ├── wsi_encoder.pth
│   ├── t1ce_flair_tumorTrue.pth
│   ├── ...
├── rna_preprocessors
│   ├── trf_0.joblib
│   ├── trf_1.joblib
│   ├── trf_2.joblib
│   ├── trf_3.joblib
│   ├── trf_4.joblib
├── get_wsi_thumbnails.py
├── run_mri_pretraining.py
├── run_wsi_pretraining.py
├── extract_brain_embeddings.py
├── preprocessing.ipynb
``````
The `scripts` folder contains all the manifest files needed to download data from the [GDC Data Portal](https://portal.gdc.cancer.gov/) (see their documentation).

For instance suppose we want data to be stocked in the `~/TCGA/GBMLGG` folder. To download RNASeq data, one can use the following command-line:
```
$ mkdir ~/TCGA/GBMLGG/raw_RNA
$ sudo /opt/gdc-client download \ 
    -d ~/TCGA/GBMLGG/raw_RNA/ \
    -m scripts/gdc_manifest_20231203_RNA_GBM_LGG.txt
```
For data pre-processing we proceeded in a similar way to [Vale-Silva et al., 2022](https://www.nature.com/articles/s41598-021-92799-4). Thus the `clinical_data.tsv` file is identical. The supplementary data (`files/supplementary_data.tsv`) contains information about the methylation and IDH status of patients.
It can be obtained using the following R code:

```
library(TCGAWorkflowData)
library(DT)
library(TCGAbiolinks)

gdc <- TCGAquery_subtype(tumor='all')
output_path <- 'files/supplementary_data.tsv'
readr::write_tsv(gdc, output_path)
```
- In the `mappings` folder, the `wsi_mapping.json` file gives the path to the corresponding WSI for each patient (see `preprocessing.ipynb` for its construction). 
 - The `BraTS2023_2017_GLI_Mapping.xlsx` file is used to link the BraTS competition MRIs to the TCGA patients, you can download it directly on the [BraTS competition homepage](https://www.synapse.org/#!Synapse:syn51156910/wiki/621282) by signing up to the competition or through The Cancer Imaging Archive (TCIA).

The `models` folder is immediately self-explanatory: it contains pretrained encoder used for the multimodal training. The maximum file size limit of the supplemental does not allow the WSI pre-trained models to be supplied. So only the MRI pre-trained model is provided.

# Preprocessing
All the pre-processing is detailed in the file `preprocessing.ipynb`
All the python files in the root are used to build the final data file used later. Their function is described in more detail in the notebook. To make a long story short:
- `get_wsi_thumbnails.py`: is used to obtain a tissue mask and a low-resolution image for each WSI.
- `run_mri_pretraining.py`: is used to train the specific encoder for MRI data.
- `run_wsi_pretraining.py`: is used to train the specific encoder for WSI data.
- `extract_brain_embeddings.py`: is used to extract embeddings from MRI data, where each volume is centered around the tumor and contains only t1ce and flair modalities (2x64x64x64).

The `rna_preprocessors` folder contains all the objects to be used to preprocess the data according to the validation split.
The script generating these files, which will be used throughout the training sessions, is present in `preprocessing.ipynb`.
# Acknowledgements
As a large part of this folder is largely inspired by [Vale-Silva's MultiSurv paper](https://www.nature.com/articles/s41598-021-92799-4), it seems important to thank him again for his very clear and reproducible work and to point it out. [Vale-Silva's GitHub repository](https://github.com/luisvalesilva/multisurv/tree/master/data)