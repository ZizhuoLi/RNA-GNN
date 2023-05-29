# RNA-GNN
Pytorch implementation of RNA-GNN for ISPRS paper "Two-view correspondence learning using graph neural network with reciprocal neighbor attention", by Zizhuo Li, Yong Ma, Xiaoguang Mei and Jiayi Ma.

This paper aims to identify reliable correspondences between two-view images and retrieving the camera motion encoded by the essential matrix. We introduce RNA-GNN, an attentional graph neural network, which can explicitly model interactions among relevant neighboring correspondences.

This repo contains the code for essential matrix estimation described in our ISPRS paper.

Welcome bugs and issues!

If you find this project useful, please cite:
```
@article{li2023rna,
  title={Two-view correspondence learning using graph neural network with reciprocal neighbor attention},
  author={Li, Zizhuo and Ma, Yong and Mei Xiaoguang and Ma, Jiayi},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  year={2023}
}
```

Part of the code is borrowed or ported from

[OANet](https://github.com/zjhthu/OANet), for training scheme,

[PointCN](https://github.com/vcg-uvic/learned-correspondence-release), for implementaion of PointCN block and geometric transformations.

Please also cite these works if you find the corresponding code useful.

## Requirements

Please use Python 3.6.15, opencv-contrib-python (3.4.1.15) and Pytorch (1.10.2). Other dependencies should be easily installed through pip or conda.

## Datasets and Pretrianed models

1. Download the YFCC100M dataset and the SUN3D dataset from the [OANet](https://github.com/zjhthu/OANet) repository.

2. Download the pretrained RNA-GNN model for outdoor scenes from [here](https://drive.google.com/file/d/1dwIj7C-F3TxGb-03zuSVSGQKVQJiIUBv/view?usp=sharing).

## Evaluation

Evaluate on the YFCC100M and SUN3D datasets with SIFT descriptors and Nearest Neighborhood (NN) matcher:
```bash
cd ./core 
python main.py --run_mode=test --model_path=../Pretrained_models/Outdoor-SIFT --use_ransac=False
python main.py --run_mode=test --data_te=../data_dump/sun3d-sift-2000-test.hdf5 --model_path=../Pretrained_models/Indoor-SIFT --use_ransac=False
```
Set `--use_ransac=True` to get results after RANSAC post-processing.

## Training

After generating dataset for YFCC100M and SUN3D, run the tranining script:
```bash
cd ./core 
bash train.sh
```

Our training scripts support multi-gpu training, which can be enabled by configure **core/train.sh** for these entries

   **CUDA_VISIBLE_DEVICES**: id of gpus to be used   
   **nproc_per_node**: number of gpus to be used

You can train the fundamental estimation model by setting `--use_fundamental=True --geo_loss_margin=0.03` and use side information by setting `--use_ratio=2 --use_mutual=2`