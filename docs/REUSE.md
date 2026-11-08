# Reuse guide — CBS2025

[README](../README.md) · [Dandan Zhang’s fork](https://github.com/dianzdd123/CBS2025) · [MSEI Lab](https://zhangrobotics.com/msei_website/)

## Get this copy

```bash
git clone https://github.com/dianzdd123/CBS2025.git
cd CBS2025
```

## Purpose and entry points

This release compares image-only and physics-informed optical-microrobot depth regression, including adaptive-grid focus features. The root README maps each experiment file to its purpose.

## Data and environment

Training expects `Image_Depth/<robot_name>/` with microscopy images and a text depth-label file. Set the selected script's `robot_name`, data root and output directory. Verify that the sorted image sequence matches the label row order; equal counts alone do not establish synchronisation. Keep adjacent frames/captures grouped appropriately when designing a new evaluation split.

The historical training scripts import PyTorch, torchvision, NumPy, pandas, Pillow, scikit-learn, OpenCV, PyWavelets, Transformers and THOP. The Colab optimisation also uses XGBoost and scikit-optimize. No tested, pinned environment was supplied. Prepare these in an isolated environment and record resolved versions. Pretrained backbones can download model weights.

## Local video feature extraction

The root feature extractor is now a normal Python script. It no longer mounts Google Drive or starts optimisation at import time. With OpenCV, NumPy, pandas and PyWavelets installed:

```bash
python local_features_extraction_for_microrobot_depth_estimation_.py /path/to/video.avi --output-dir ./results/features
```

It writes `<video-stem>_focus_features.csv`, one row per contour-qualified frame and one column per grid-cell/focus-function combination. Input videos need distinct stems; existing output files are not overwritten. Frames without an accepted contour are omitted, so row indices are **not** original frame timestamps. Align retained samples with depth annotations explicitly before training.

The original interactive export, including Google Drive paths, uploads, depth joining and Bayesian XGBoost search, is preserved in [notebooks/local_features_colab.ipynb](../notebooks/local_features_colab.ipynb). Review its paths and cell sequence before running it; it has not been executed in this maintenance pass.

## Training and evaluation

Execute a selected experiment script only after configuring its top-level settings and data. For example, `python depth_baseline_data_size.py` launches the configured image baseline experiments; it has no argparse configuration interface. Other root scripts launch their own configured physics-informed variants. These programs still perform data setup/training at module level, so do not import them as utility libraries.

Validation and test loaders now retain their final partial batches. Scalar regression predictions preserve the batch dimension even for a single sample. Training logs report the sample-weighted mean loss rather than resetting the accumulator to zero. Training retains `drop_last=True` for the existing BatchNorm-dependent architectures; a dataset too small to produce a training batch now raises a clear error.

These changes can alter reported evaluation metrics because previously omitted samples are included. Compare runs with the commit SHA and evaluated sample counts recorded. Checkpoints and result CSV paths are declared near the top of each script.

## Validation scope

The repository check verifies Python syntax, notebook containers and local navigation. Four focused regression tests also exercise the actual experiment functions with array-backed Torch fixtures: final single-sample evaluation, weighted loss accounting, empty training loaders and unsigned-image arithmetic in the Brenner focus measure. No training, video decoding, optimisation or paper-metric reproduction is claimed. The preserved Colab notebook still requires its original interactive environment and data preparation.

## Local quality checks

Run `python scripts/check_repository.py` from a Git checkout with Python 3.10+ and Git installed. It parses Python, JSON/notebook containers and links in the navigation documents without importing research modules. Notebook cells and remote links are not executed or tested.

## Attribution and reuse terms

This is a fork of [LannWei/CBS2025](https://github.com/LannWei/CBS2025). Original paper authorship, upstream citations and notices remain authoritative. Cite the associated work and identify this fork’s commit when reporting modifications.

No repository-wide licence file is present in this snapshot. Obtain the authors’ reuse terms before assuming permission to redistribute code, datasets or checkpoints. This documentation update does not relicense the work.
