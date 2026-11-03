# Brain MRI Segmentation with nnU-Net

Pixel-level segmentation of Lower-Grade Glioma (LGG) tumors in 2D brain MRI slices, using a custom **2D U-Net** built with **nnU-Net** preprocessing and optimization principles (PyTorch).

> The full project report (in Turkish) is included in this repository.

---

## Results

The best checkpoint (epoch 24) was evaluated at the pixel level on a held-out test set of **17 unseen patients (561 slices)**:

| Metric | Test score |
| --- | --- |
| Precision | **0.8785** |
| Recall | **0.8064** |
| F1 score (Dice) | **0.8409** |
| IoU (Jaccard) | **0.7255** |
| Pixel accuracy | 0.9959 |

- Best validation Dice: **0.819** (epoch 24), validation IoU ≈ 0.72
- False positive rate: **0.15%** of healthy pixels; false negative rate: **19.36%** of tumor pixels

> Pixel accuracy is misleading here: in tumor-containing slices, tumors cover only **2.95%** of the image on average. A model that predicts "background" everywhere would still score ~97%. Dice and IoU are the metrics that matter for this task.

---

## Problem

LGGs are slow-growing brain tumors that infiltrate healthy tissue. Delineating their boundaries on MRI is essential for surgery, radiotherapy planning and follow-up. Doing this by hand is slow, and different experts draw different boundaries (inter-observer variability).

- **Input:** 2D brain MRI slice, 256×256×3
- **Output:** 256×256×1 binary mask (0 = healthy tissue / background, 1 = tumor / FLAIR abnormality)

---

## Dataset

**LGG MRI Segmentation** (Mateusz Buda et al.): TCIA / TCGA LGG collections, available on Kaggle.
🔗 https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/

- **3,929** MRI slices (`.tif`) from **110** patients, each with an expert-verified binary mask (`_mask.tif`)
- **3 channels:** pre-contrast, FLAIR and post-contrast. Missing sequences (9 patients lack post-contrast, 6 lack pre-contrast) are filled with copies of FLAIR.
- **Class imbalance:**
  - Slice level: 65.1% of slices contain no tumor (2,556 slices) and 34.9% contain tumor (1,373 slices).
  - Pixel level: in tumor slices, tumor pixels cover only 2.95% of the image on average.

The notebook downloads the dataset automatically with `kagglehub`. It is not stored in this repository.

---

## Method

**Preprocessing**
- **Patient-level split (70 / 15 / 15):** 77 / 16 / 17 patients (2,801 / 567 / 561 slices). The three sets share no patients, which prevents data leakage between them.
- **Augmentation (train only, Albumentations):** horizontal and vertical flips (p = 0.5) and ±15° rotation (p = 0.5). Each transform is applied to the image and its mask together.
- **nnU-Net z-score normalization:** each channel of each slice is normalized with its own mean and standard deviation: `(I − μ) / (σ + 1e-8)`.

**Model: 2D U-Net**
- Encoder 64 → 128 → 256 → 512, bottleneck 1024, symmetric decoder with `ConvTranspose2d` upsampling and skip connections
- `DoubleConv` block: 2 × (Conv 3×3, `bias=False` → BatchNorm → LeakyReLU(0.01))
- 1×1 output convolution that produces raw logits (the sigmoid is applied inside the loss)

**Training**

| Setting | Value |
| --- | --- |
| Loss | BCE + Soft Dice (`BCEDiceLoss`) |
| Optimizer | AdamW (lr = 1e-4, weight decay = 1e-4) |
| Scheduler | ReduceLROnPlateau (factor = 0.5, patience = 3) |
| Epochs / batch size | 25 / 16 |
| Checkpointing | Best validation Dice |
| Decision threshold | p > 0.5 |

---

## Repository Structure

```
.
├── brain_mri_segmentation_with_nnunet.ipynb   # End-to-end notebook: EDA, preprocessing, model, training, evaluation
├── Project Report.pdf                         # Project report (Turkish)
├── history.json                               # Per-epoch train/val loss, Dice and IoU (25 epochs)
└── README.md
```

**Model weights:** `best_unet_model.pth` (trained U-Net weights, best validation Dice at epoch 24, ~118 MB) is too large to keep in the repository. Download it from the [**Releases**](https://github.com/<uturkmen24>/<brain-mri-segmentation-with-nnunet>/releases/latest) page.

---

## How to Run Locally

### 1. Clone the repository and download the model

```bash
git clone https://github.com/<uturkmen24>/<brain-mri-segmentation-with-nnunet>.git
cd <repo-name>
```

Download `best_unet_model.pth` from the [Releases](https://github.com/<uturkmen24>/<brain-mri-segmentation-with-nnunet>/releases/latest) page and place it in the repository folder, next to `history.json`. You need it only for option A in step 4.

### 2. Create an environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install torch albumentations opencv-python kagglehub pandas numpy \
            matplotlib seaborn scikit-learn tqdm ipywidgets jupyter
```

For GPU training, install the CUDA build of PyTorch that matches your system (see [pytorch.org](https://pytorch.org/get-started/locally/)). The notebook falls back to CPU automatically.

### 3. Point the notebook to the local files

The notebook was developed on Google Colab and saves its outputs to Google Drive. In the **Training & Validation Loop** cell, change the two paths to the files in this repository:

```python
history_save_path = 'history.json'
model_save_path   = 'best_unet_model.pth'
```

### 4. Run the notebook

```bash
jupyter notebook brain_mri_segmentation_with_nnunet.ipynb
```

Run all cells. The first run downloads the dataset through `kagglehub`; you may need to set up your [Kaggle API credentials](https://www.kaggle.com/docs/api) for this.

There are two ways to use the notebook:

| Option | What to do | What happens |
| --- | --- | --- |
| **A. Reproduce the results (no training)** | Download `best_unet_model.pth` from Releases and put it next to `history.json` | The notebook finds both files, skips training and loads the saved weights and history. The training curves, test metrics, confusion matrix and error maps are then regenerated. |
| **B. Train from scratch** | Do not download the model, and move or rename `history.json` | The model trains for 25 epochs, writing `history.json` after each epoch and saving `best_unet_model.pth` whenever validation Dice improves. A GPU is strongly recommended. |

> ⚠️ Training is skipped only when **both** `history.json` and `best_unet_model.pth` are present. If you run the notebook without downloading the model, it trains from scratch and **overwrites `history.json`**.

**Windows tip:** if the DataLoader raises a multiprocessing error, set `num_workers=0` in the `DataLoader` cell.

---

## Limitations & Future Work

- **Missed tumor margins (false negatives):** infiltrative, poorly defined LGG boundaries often get probabilities just below the fixed 0.5 threshold.
- **Over-segmentation (false positives):** the model tends to fill narrow sulcal gaps and extend into nearby hyperintense edema.
- **No 3D context:** each slice is processed independently.

Possible improvements include a 2.5D or 3D U-Net (adjacent slices as input), a boundary/surface loss, tuning the threshold on the validation precision-recall curve, connected component post-processing and test-time augmentation (TTA).

---

## Author

**Umut Türkmen**
