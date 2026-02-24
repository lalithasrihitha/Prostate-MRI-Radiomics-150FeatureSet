

# Prostate MRI Radiomics – 150 Feature Set

This repository provides a reproducible 3D radiomics feature extraction pipeline for prostate T2-weighted MRI using gland segmentation masks. The pipeline generates a standardized dataset of 150 quantitative radiomic features per patient.

---

## Overview

- Modality: T2-weighted Prostate MRI  
- Segmentation: Prostate gland mask  
- Extraction Dimension: 3D  
- Bin Width: 25  
- Total Features per Patient: 150  

Feature Composition:
- 107 Original image radiomic features  
- 43 LoG-filtered radiomic features (sigma = 1.0)  

The feature dimensionality is controlled to produce a consistent and model-ready feature matrix.

---

## Processing Workflow

1. Load T2-weighted MRI (NIfTI format)
2. Normalize image intensity
3. Binarize gland segmentation mask
4. Resample mask to match MRI geometry (nearest neighbor)
5. Extract Original radiomics features
6. Extract LoG features (sigma=1.0)
7. Generate structured CSV output (patients × 150 features)

---

## Extracted Feature Classes

- First Order Statistics  
- Shape (3D)  
- GLCM  
- GLRLM  
- GLSZM  
- GLDM  
- NGTDM  

---

## Output

The pipeline produces:

pyradiomics_150_features.csv  

Format:
N patients × 150 features  

Each row corresponds to one patient.  
Each column represents a quantitative radiomic biomarker.

---

## Technologies Used

- Python  
- PyRadiomics  
- SimpleITK  
- NumPy  
- Pandas  

---

## Use Case

The extracted feature set is designed for:

- Prostate cancer classification  
- Risk stratification modeling  
- Machine learning pipelines (XGBoost, Random Forest, etc.)  
- Imaging biomarker research  

---

## Data Availability

MRI data and extracted feature files are not included due to institutional research restrictions.

---

