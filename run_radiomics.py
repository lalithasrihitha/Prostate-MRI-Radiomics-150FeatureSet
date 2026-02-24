import SimpleITK as sitk
import pandas as pd
import numpy as np
import numbers
import logging
import sys
from pathlib import Path
from radiomics import featureextractor

logging.getLogger("radiomics").setLevel(logging.ERROR)
BASE_DIR = Path("data/raw_mri")
OUT_CSV  = Path("outputs/pyradiomics_150_features.csv")

# Original extractor
ext_orig = featureextractor.RadiomicsFeatureExtractor()
ext_orig.settings["binWidth"] = 25
ext_orig.enableImageTypeByName("Original")
ext_orig.enableAllFeatures()

# LoG extractor (sigma=1.0 only)
ext_log = featureextractor.RadiomicsFeatureExtractor()
ext_log.settings["binWidth"] = 25
ext_log.enableImageTypeByName("LoG", customArgs={"sigma": [1.0]})
ext_log.enableAllFeatures()

def to_float(v):
    if isinstance(v, numbers.Number):
        return float(v)
    if isinstance(v, np.ndarray) and v.size == 1:
        return float(v.item())
    return None

def clean_features(feat_dict):
    return {
        k: to_float(v)
        for k, v in feat_dict.items()
        if not k.startswith("diagnostics")
        and to_float(v) is not None
        and np.isfinite(to_float(v))
    }

patient_dirs = sorted([d for d in BASE_DIR.iterdir() if d.is_dir()])
print(f"Found {len(patient_dirs)} patient folders\n", flush=True)

if len(patient_dirs) == 0:
    print(f"ERROR: No patient folders found in {BASE_DIR}")
    sys.exit(1)

orig_cols = None
log_cols  = None
rows = []

for i, patient_dir in enumerate(patient_dirs, 1):
    case_id   = patient_dir.name
    t2_path   = patient_dir / f"{case_id}_t2w.nii.gz"
    mask_path = patient_dir / f"{case_id}_gland.nii.gz"

    print(f"[{i}/{len(patient_dirs)}] Processing {case_id}...", end=" ", flush=True)

    if not t2_path.exists():
        print("SKIPPED - t2w not found", flush=True)
        continue
    if not mask_path.exists():
        print("SKIPPED - mask not found", flush=True)
        continue

    try:
        # Load image
        image    = sitk.Cast(sitk.ReadImage(str(t2_path)), sitk.sitkFloat32)
        image    = sitk.Normalize(image)

        # Load and resample mask
        mask     = sitk.ReadImage(str(mask_path))
        mask_bin = sitk.BinaryThreshold(mask, 1e-12, 1e9, 1, 0)
        mask_bin = sitk.Cast(mask_bin, sitk.sitkUInt8)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        resampler.SetOutputPixelType(sitk.sitkUInt8)
        mask_bin = resampler.Execute(mask_bin)

        # Extract Original features
        feat_orig  = ext_orig.execute(image, mask_bin, label=1)
        clean_orig = clean_features(feat_orig)

        # Lock column order on first patient
        if orig_cols is None:
            orig_cols = sorted(clean_orig.keys())
            print(f"\nOriginal features: {len(orig_cols)}", flush=True)

        row = {k: clean_orig.get(k, np.nan) for k in orig_cols}

        # Add LoG features to reach exactly 150
        need = 150 - len(orig_cols)
        if need > 0:
            feat_log  = ext_log.execute(image, mask_bin, label=1)
            clean_log = clean_features(feat_log)

            if log_cols is None:
                log_cols = sorted(clean_log.keys())[:need]
                print(f"LoG features added: {len(log_cols)}", flush=True)
                print(f"Total will be: {len(orig_cols) + len(log_cols)}", flush=True)

            for k in log_cols:
                row[k] = clean_log.get(k, np.nan)

        row["CaseID"] = case_id
        rows.append(row)
        print(f"done ({len(row)-1} features)", flush=True)

    except Exception as e:
        print(f"ERROR - {e}", flush=True)
        continue

if len(rows) == 0:
    print("\nERROR: No cases processed successfully.")
    sys.exit(1)

df = pd.DataFrame(rows).set_index("CaseID")
print(f"\nFinal shape before assert: {df.shape}", flush=True)

assert df.shape[1] == 150, f"Expected 150 features but got {df.shape[1]}"

df.to_csv(OUT_CSV)
print(f"\n{'='*50}")
print(f"DONE!")
print(f"Saved to:    {OUT_CSV}")
print(f"Final shape: {df.shape}  ({df.shape[0]} patients x {df.shape[1]} features)")
print(f"Sample columns: {list(df.columns[:10])}")
