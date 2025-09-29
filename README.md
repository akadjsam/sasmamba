# SasMamba: A Lightweight Structure-Aware Stride State Space Model for 3D Human Pose Estimation

This is the official PyTorch implementation of our paper:  *SasMamba: A Lightweight Structure-Aware Stride State Space Model for 3D Human Pose Estimation*

## 💡Environment

The project is developed under the following environment:

- PyTorch  2.0.0
  
  Python  3.8(ubuntu20.04)
  
  CUDA  11.8

For installation of selective_scan:

```
cd model/selective_scan && pip install -e .
```

## 🐳 Dataset

### Human3.6M

#### Preprocessing

We follow the previous state-of-the-art method [MotionBERT](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md) for dataset setup. Download the [MotionBERT](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md)'s preprocessed H3.6M data [here](https://1drv.ms/u/s!AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ) and unzip it to 'data/motion3d'.

**For our model with T = 243**:

```text
python h36m.py  --n-frames 243
```

### MPI-INF-3DHP

#### Preprocessing

Please refer to [P-STMO](https://github.com/paTRICK-swk/P-STMO#mpi-inf-3dhp) for dataset setup. After preprocessing, the generated .npz files (`data_train_3dhp.npz` and `data_test_3dhp.npz`) should be located at `data/motion3d` directory.

#### The data dic structure like:

```
.
├── H36M-263
│ ├── train
│ ├── test
│── data_test_3dhp.npz
│── data_train_3dhp.npz
```

## ✨ Training

After dataset preparation, you can train the model as follows:

### Human3.6M

You can train Human3.6M with the following command:

```
python train.py --config configs/strideposemamba_h36m_243.yaml
```

### MPI-INF-3DHP

You can train MPI-INF-3DHP with the following command:

```
python train_3dhp.py --config configs/stridemamba_mpi_xx.yaml
```

## 🚅 Evaluation

| Dataset      | frames | Checkpoint  |
| ------------ | ------ | ----------- |
| Human3.6M    | 243    | coming soon |
| Human3.6M(L) | 243    | coming soon |
| MPI-INF-3DHP | 27     | coming soon |
| MPI-INF-3DHP | 81     | coming soon |

After downloading the weight from table above, you can evaluate Human3.6M models by:

```
python train.py --eval-only --checkpoint <CHECKPOINT-DIRECTORY> --checkpoint-file <CHECKPOINT-FILE-NAME> --config <PATH-TO-CONFIG>
```

## 👀 Visualization

- Download the yolo weights at : open later

- put the weiths in : `demo/lib/checkpoint`

- `vis_h36data_sample.py` used for h360 data sample visualization. 

- `vis_video_hrnet_sample.py` used for videos for wild scense.

## ✏️ Citation

## 👍 Acknowledgement

Our code is extended from the following repositories. We thank the authors for releasing the codes. 

- [MixSTE](https://github.com/JinluZhang1126/MixSTE)
- [StridedTransformer-Pose3D](https://github.com/Vegetebird/StridedTransformer-Pose3D)
- [PoseMamba](https://github.com/nankingjing/PoseMamba)
- [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer)
- [HGMamba](https://github.com/HuCui2022/HGMamba)

## 🔒 Licence

This project is licensed under the terms of the MIT license.
