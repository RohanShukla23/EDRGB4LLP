# DSEC Preprocessing Workflow (Windows → GCP)

## Overview
This workflow processes DSEC event camera data locally on Windows, then uploads to GCP bucket for cloud-based training.

## Directory Structure (After Processing)
```
EDRGB4LLP/
├── raw/                          # Raw DSEC data (auto-downloaded)
│   └── DSEC/
│       ├── train/
│       └── test/
├── processed/
│   └── v1/
│       ├── sequences/
│       │   ├── dsec_train/
│       │   │   ├── rgb/          # JPEG images
│       │   │   ├── events/       # Voxel grid .npz files
│       │   │   ├── labels/       # JSON label files
│       │   │   └── meta/         # Metadata JSON
│       │   └── dsec_test/
│       │       └── ...
│       └── splits/
│           ├── train.jsonl       # Training split index
│           └── test.jsonl        # Test split index
└── preprocess_dsec.py
```

## Step-by-Step Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: You may need to install PyTorch separately for your system:
```bash
# For Windows with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For Windows CPU only
pip install torch torchvision
```

### 2. Process DSEC Dataset (Local)

**Small test run first** (processes only 5 samples):
```bash
python preprocess_dsec.py --split train --max-samples 5
```

**Full training set** (may take hours + download 20+ GB):
```bash
python preprocess_dsec.py --split train --max-samples -1
```

**Full test set**:
```bash
python preprocess_dsec.py --split test --max-samples -1
```

**Monitor progress**: 
- Watch `./processed/v1/sequences/dsec_train/rgb/` folder for output files
- Check console for progress: `[info] processed 100/1000`

### 3. Upload to GCP Bucket

#### First Time Setup
1. **Install Google Cloud SDK** (if not already installed):
   - Download: https://cloud.google.com/sdk/docs/install
   - Or use Cloud Shell

2. **Authenticate**:
   ```bash
   gcloud auth application-default login
   ```
   This opens browser to login with your Google account.

3. **Verify access** to your bucket:
   ```bash
   gsutil ls gs://cmpm-bucket/
   ```

#### Upload Processed Data

**Dry run** (see what will be uploaded without uploading):
```bash
python upload_to_gcp.py --dry-run
```

**Upload training data**:
```bash
python upload_to_gcp.py --local-root ./processed/v1 --gcp-bucket cmpm-bucket --gcp-prefix event-rgb/processed/v1
```

**Features**:
- ✅ Skips files that already exist with same size (resume interrupted uploads)
- ✅ Progress bar with upload statistics
- ✅ Preserves directory structure
- ✅ Handles Windows paths correctly

**View uploaded data**:
https://console.cloud.google.com/storage/browser/cmpm-bucket/event-rgb/processed/v1

## Configuration

Edit `config.yaml` to change:
- **Resolution**: `H: 320, W: 320` (default)
- **Event bins**: `bins: 4` (temporal slices)
- **Event window**: `window_ms: 10` (10 milliseconds)
- **Paths**: `bucket_root` and `raw_local_root`

## Troubleshooting

### Issue: "No module named 'google.cloud'"
**Fix**: 
```bash
pip install google-cloud-storage
```

### Issue: "Permission denied" when uploading to GCS
**Fix**: Re-authenticate
```bash
gcloud auth application-default login
```

### Issue: DSEC download fails
**Fix**: 
- Check internet connection
- The download is large (20+ GB), be patient
- If interrupted, restart - Tonic will resume download

### Issue: Out of disk space
**Fix**:
- Raw DSEC data: ~20-30 GB
- Processed data: ~10-20 GB
- Ensure you have 50+ GB free space

### Issue: Out of memory during processing
**Fix**: Reduce `--max-samples` to process in smaller batches
```bash
# Process 100 at a time
python preprocess_dsec.py --split train --max-samples 100
```

Then manually edit `train.jsonl` to append subsequent runs.

## Expected Processing Times

| Stage | Samples | Time (estimate) |
|-------|---------|-----------------|
| Download DSEC | - | 1-3 hours |
| Process 50 samples | 50 | 5-10 minutes |
| Process full train | ~3000 | 2-4 hours |
| Upload to GCP | ~3000 | 30-60 minutes |

*Times vary based on hardware and internet speed*

## Next Steps

After uploading to GCP:
1. Use GCP VM or Colab to train models
2. Mount bucket: `gcsfuse cmpm-bucket /mnt/bucket`
3. Or use `gs://cmpm-bucket/event-rgb/processed/v1` paths directly in PyTorch

## File Formats

- **RGB**: JPEG images (320x320x3), 95% quality
- **Events**: NPZ compressed arrays, shape (4, 320, 320), float32
- **Labels**: JSON with `{"boxes": [], "classes": []}` (currently empty placeholders)
- **Meta**: JSON with timestamps, config info
- **Splits**: JSONL (one JSON object per line) with paths to all files
