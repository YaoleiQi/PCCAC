<h1 align="center">Rethinking the Detail-Preserved Completion of Complex Tubular Structures based on Point Cloud: a Dataset and a Benchmark</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2308.14383-b31b1b.svg)](https://arxiv.org/abs/2508.17658)
![GitHub repo size](https://img.shields.io/github/repo-size/YaoleiQi/PCCAC)
[![Dataset_download](https://img.shields.io/badge/Dataset-Download-green)](https://huggingface.co/datasets/Ryan710/PC-CAC/tree/main)
[![Checkpoint_download](https://img.shields.io/badge/Checkpoint-Download-orange)](https://drive.google.com/file/d/1KA7BT2xBvI-Od7zXrXPvJVlv_FJ0j15E/view?usp=drive_link)
![GitHub forks](https://img.shields.io/github/forks/YaoleiQi/PCCAC)
[![GitHub](https://img.shields.io/github/stars/YaoleiQi/PCCAC)](https://github.com/YaoleiQi/PCCAC)

<!--
> **Download** PC-CAC datasets: [Baidu](https://pan.baidu.com/s/10-W0Crs0MXU2dYiiOY_RfA?pwd=7100) 7100 | [Google](https://drive.google.com/file/d/1XJ5Ks_T2aCU8tZVu6_d3n9_MbO4vaavD/view?usp=sharing)
-->

Yaolei Qi<sup>#</sup>, Yikai Yang<sup>#</sup>, Wenbo Peng, Shumei Miao, Yutao Hu<sup>✉</sup>, Guanyu Yang<sup>✉</sup>, Rethinking the Detail-Preserved Completion of Complex Tubular Structures based on Point Cloud: a Dataset and a Benchmark, MedIA, 2026


## 📢 News
- [**June, 2026**] We are updating the code.
- [**June, 2026**] We have released the **Dataset**: [**PC-CAC**](https://huggingface.co/datasets/Ryan710/PC-CAC/tree/main).
- [**June, 2026**] The paper has been accepted by **Medical Image Analysis** (MedIA) [IF:14].
- [**Aug, 2025**] The paper is available on arXiv.


## 📊 Visualization
<div align="center"><img src="Fig/Results.png" alt="results" style="zoom:60%;" /></div>


## 🔑 Key Innovation
- **The first point cloud-based tubular structure reconnection dataset**: To our best knowledge, we build the first point cloud-based coronary artery (PC-CAC) dataset from clinical data. This dataset will be open-sourced, offering a new perspective for tubular structure reconnection and fostering advancements in this field.
- **A novel exploration and high-performing baseline**: Our work represents the first attempt to explore tubular structure reconnection from a point cloud perspective. We propose a baseline designed for accurately reconnecting fractured tubular structures, comprising a detail-preserved feature extractor, a multiple dense refinement strategy, and a global-to-local loss function. These methods cooperate to enhance detail preservation and effectively handle hard-to-represent regions.
- **A sufficient evaluation with experiments**: To objectively evaluate our approach, experiments are conducted on our PC-CAC dataset and two public datasets. Experimental results show that our method achieves state-of-the-art performance across multiple datasets.

## 🚀 Motivation
<div align="center"><img src="Fig/Motivation_reb.png" alt="results" style="zoom:60%;" /></div>

## ⚡ Abstract
Complex tubular structures are essential in medical imaging and computer-assisted diagnosis, where their integrity enhances anatomical visualization and lesion detection. However, existing segmentation algorithms struggle with structural discontinuities, particularly in severe clinical cases such as coronary artery stenosis and vessel occlusions, which leads to undesired discontinuity and compromising downstream diagnostic accuracy. Therefore, it is imperative to reconnect discontinuous structures to ensure their completeness. In this study, we explore the tubular structure reconnection from a unique point cloud perspective for the first time and establish a novel Point Cloud-based Coronary Artery Completion (PC-CAC) dataset, which is derived from real clinical data. This dataset provides a novel benchmark for tubular structure reconnection. Additionally, we propose TSRNet, a novel Tubular Structure Reconnection Network that integrates a detail-preservated feature extractor, a multiple dense refinement strategy, and a global-to-local loss function to ensure accurate reconnection while maintaining structural integrity. Comprehensive experiments on our PC-CAC and two additional public datasets (PC-ImageCAS and PC-PTR) demonstrate that our method consistently outperforms state-of-the-art approaches across multiple evaluation metrics, setting a new benchmark for point cloud-based tubular structure reconstruction.


## 🔍 How to perform the voxel-to-point cloud conversion process.
<div align="center"><img src="Fig/A.png" alt="results" style="zoom:60%;" /></div>

## 📊 Dataset Overview

### a) The details of two publicly available datasets in our task

| Name        | Target Dataset | Train/Val/Test     | Detail Information                                                                                                                                     | Pre-processing                                                                                             |
|-------------|----------------|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| PC-ImageCAS | ImageCAS [[1]](#ref1)  | 700/100/200<br>5600/800/1600 | 1. Scanner: Siemens 128-slice dual-source  <br> 2. Planar resolution: 0.29 ~ 0.43 mm²  <br> 3. Slice thickness: 0.25 ~ 0.5 mm  <br> 4. x/y-size: 512 voxels, z-size: ~206 ~ 275 voxels | 1. Resample the resolution to 1 mm³  <br> 2. Normalize via `max(min(0,x),2048)/2048` <br> 3. Obtain segmentation results <br> 4. Extract **aorta** and main **coronary** branches <br> 5. Generate point cloud based on surface of **aorta** <br> 6. Generate point cloud based on centerline of **coronary** |
| PC-PTR      | PTR [[2]](#ref2)      | 599/80/160<br>4472/640/1280  | 1. Scan from multiple medical centers  <br> 2. Resolution: already processed to 1 mm³  <br> 3. x/y-size: 512 voxels, z-size: 177 ~ 798 voxels           | Generate point cloud based on centerline of vessels                                                        |

---

### b) The details of our proposed dataset from real clinical data

| Name   | Target Dataset | Train/Val/Test     | Detail Information                                                                                                         | Pre-processing                                                                                             |
|--------|----------------|--------------------|----------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| PC-CAC | PC-CAC         | 300/40/87<br>2400/320/696 | 1. Scanner: SOMATOM Definition Flash <br> 2. x/y-resolution: 0.25 ~ 0.57 mm/voxel  <br> 3. Slice thickness: 0.75 ~ 3 mm/voxel  <br> 4. x/y-size: 512 voxels, z-size: 128 ~ 994 voxels | 1. Resample the resolution to 1 mm³  <br> 2. Normalize via `max(min(0,x),2048)/2048` <br> 3. Obtain segmentation results <br> 4. Extract **aorta** and main **coronary** branches <br> 5. Generate point cloud based on surface of **aorta** <br> 6. Generate point cloud based on centerline of **coronary** |

---


### 📌 Notes
> To ensure a diverse and challenging reconstruction task, each patient generates 8 distinct input cases, capturing a wide range of conditions.

<a name="ref1">[1]</a> ImageCAS: [https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT)  
<a name="ref2">[2]</a> PTR: [https://github.com/M3DV/pulmonary-tree-repairing](https://github.com/M3DV/pulmonary-tree-repairing)

## Dataset
**Download** PC-CAC datasets: [Baidu](https://pan.baidu.com/s/10-W0Crs0MXU2dYiiOY_RfA?pwd=7100) | [Google](https://drive.google.com/file/d/1XJ5Ks_T2aCU8tZVu6_d3n9_MbO4vaavD/view?usp=sharing) | [Huggingface](https://huggingface.co/datasets/Ryan710/PC-CAC/tree/main)

``` 📂 PC_CAC/
├── Train/               # Train dataset
  ├── input_broken/      # Fractured vessel input (input)
    ├── 1/               # Patients ID = 1 (input)
      ├── 0.ply          # Fractured case 0
      ├── 1.ply          # Fractured case 1
      ├── ...
      └── 7.ply          # Fractured case 7 (simulate 8 types)
    ├── 2/
    ├── ...
    └── 300/
  └── lable_complete/    # Fully connected vessel (ground truth)
    ├── 1.ply            # Patients ID = 1 (ground truth)
    ├── 2.ply
    ├── ...
    └── 300.ply
├── Test/                # Test dataset
└── Val/                 # Validation dataset
```

## 🛠️ Usage
---
## 1. Environment Setup
Create a Python environment and install the required dependencies:
```
conda create -n tsrnet python=3.8
conda activate tsrnet
```
Install other dependencies:
```
pip install torch torchvision torchaudio
```

If your project uses custom CUDA extensions such as Chamfer Distance, Earth Mover's Distance, or PointNet++ operators, please compile them before training or testing.
For example:
```
cd extensions/chamfer_distance
python setup.py install
cd ../earth_movers_distance
python setup.py install
If pointnet2_ops is used, please also install or compile it according to your environment.
```
## 2. Dataset Preparation
The dataset should be organized as follows:
```
CAS/
├── train/
│   ├── complete/
│   │   ├── 1.ply
│   │   ├── 2.ply
│   │   └── ...
│   └── partial/
│       ├── 1/
│       │   ├── 0.ply
│       │   ├── 1.ply
│       │   └── ...
│       ├── 2/
│       └── ...
├── val/
│   ├── complete/
│   └── partial/
├── test/
│   ├── complete/
│   └── partial/
├── train.list
├── val.list
└── test.list
```
## 3. Training
To train TSRNet from scratch, run:
```
python train.py \
  --dataroot CAS \
  --exp_name TSRNet \
  --category all \
  --batch_size 1 \
  --epochs 400 \
  --lr 0.0001 \
  --device cuda:0
```
The training logs and checkpoints will be saved to:
```
log/TSRNet/all/
```

The best model based on L1 Chamfer Distance will be saved as:
```
log/TSRNet/all/checkpoints/best_all_l1_cd.pth
```
If you want to resume training from a pretrained checkpoint:
```
python train.py \
  --dataroot CAS \
  --ckpt_path log/TSRNet/all/checkpoints/best_all_l1_cd.pth \
  --device cuda:0
```

## 4. Evaluation
To evaluate the trained TSRNet model:

```
python test.py \
  --dataroot CAS \
  --ckpt_path log/TSRNet/all/checkpoints/best_all_l1_cd.pth \
  --category all \
  --batch_size 1 \
  --device cuda:0
```
The script reports metrics including:

- L1 Chamfer Distance
- L2 Chamfer Distance
- F-score
- Density-aware Chamfer Distance, DCD

If you want to evaluate EMD separately:
```
python test.py \
  --dataroot CAS \
  --ckpt_path log/TSRNet/all/checkpoints/best_all_l1_cd.pth \
  --emd \
  --device cuda:0
```

## 5. Saving Test Results
By default, the test script saves reconstructed point clouds and visualization images to:

```
test_res/TSRNet/
The saved results include:
```

```
test_res/TSRNet/coronary/
├── image/
├── output_input/
├── output_fps/
├── output_coarse/
├── output_res/
└── output_gt/
```
If you do not want to save test results, use:

```
python test.py \
  --dataroot CAS \
  --ckpt_path log/TSRNet/all/checkpoints/best_all_l1_cd.pth \
  --no_save
```

## 6. Inference on a Single Point Cloud
You can also load TSRNet manually and run inference on a single point cloud:
```
import torch
import open3d as o3d
import numpy as np
from models import TSRNet
def read_point_cloud(path):
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points, dtype=np.float32)
def save_point_cloud(path, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd, write_ascii=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = TSRNet().to(device)
model.load_state_dict(
    torch.load("log/TSRNet/all/checkpoints/best_all_l1_cd.pth", map_location=device)
)
model.eval()
points = read_point_cloud("example_partial.ply")
points = torch.from_numpy(points).unsqueeze(0).to(device)
with torch.no_grad():
    coarse, fine, output = model(points)
output_points = output[0].cpu().numpy()
save_point_cloud("reconstructed.ply", output_points)
```
The final completed point cloud is output, while coarse and fine are intermediate reconstruction results.

## 7. Model Output
TSRNet returns three point clouds:

```
coarse, fine, output = model(partial)
```
where:

- coarse: FPS-sampled coarse point cloud
- fine: first-stage refined point cloud
- output: final completed point cloud
- 
The input shape should be:

```
(B, N, 3)
```
and the output shape is typically:

```
coarse: (B, 1024, 3)
fine:   (B, 2048, 3)
output: (B, 4096, 3)
```


## Citation
```
@article{qi2026rethinking,
  title={Rethinking the detail-preserved completion of complex tubular structures based on point cloud: A dataset and a benchmark},
  author={Qi, Yaolei and Yang, Yikai and Peng, Wenbo and Miao, Shumei and Hu, Yutao and Yang, Guanyu},
  journal={Medical Image Analysis},
  pages={104179},
  year={2026},
  publisher={Elsevier}
}
```

