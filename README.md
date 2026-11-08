# Physics-Informed Optical Microrobot Depth Estimation

**MSEI Lab:** [Website](https://zhangrobotics.com/msei_website/) · [Website on GitHub](https://github.com/dianzdd123/msei_website)

**Repository:** [dianzdd123/CBS2025](https://github.com/dianzdd123/CBS2025)

Research scripts associated with **Physics-Informed Machine Learning with Adaptive Grids for Optical Microrobot Depth Estimation**, by Lan Wei, Lou Genoud and Dandan Zhang (IEEE CBS 2025).

**Upstream attribution:** Forked from [LannWei/CBS2025](https://github.com/LannWei/CBS2025). Original authorship and publication credit are retained. Changes in this copy apply to this fork.

## Get this copy

```bash
git clone https://github.com/dianzdd123/CBS2025.git
cd CBS2025
```

## File guide

| File | Purpose |
|---|---|
| [piml_depth.py](piml_depth.py) | Depth-estimation experiment using local image focus features |
| [piml_depth_adapt.py](piml_depth_adapt.py) | Adaptive-grid feature extraction and depth-regression experiment |
| [piml_depth_adapt_data_size.py](piml_depth_adapt_data_size.py) | Adaptive-grid experiment with different training-data fractions |
| [depth_baseline_data_size.py](depth_baseline_data_size.py) | Image-based depth baselines with different training-data fractions |
| [depth_model_zoo.py](depth_model_zoo.py) | CNN and pretrained vision backbones for depth regression |
| [local_features_extraction_for_microrobot_depth_estimation_.py](local_features_extraction_for_microrobot_depth_estimation_.py) | Colab-exported feature-extraction workflow |
| [Draw.ipynb](Draw.ipynb) | Plotting notebook |

## Before running

The dataset and trained checkpoints are not included. The training scripts expect microscopy images and depth labels under `Image_Depth/<robot_name>/`. Set the robot name, input directory and output paths in the selected script. Check that the sorted image list aligns with the depth-label text file before training.

The scripts import PyTorch, torchvision, NumPy, pandas, Pillow, scikit-learn, OpenCV, PyWavelets, Transformers and THOP. No pinned environment or dependency manifest is provided in this release. Model initialisation may download pretrained weights. Training is not validated by this documentation update.

The feature-extraction `.py` file contains notebook shell syntax (`!pip install`) and mounts Google Drive. It is a Colab export, not a directly runnable local Python script; adapt its cells and paths for your environment first.

## Citation and reuse

Please cite the [associated paper](https://arxiv.org/abs/2509.02343) when using this research. This checkout has no repository-wide licence file; consult the authors about reuse terms before redistribution.
