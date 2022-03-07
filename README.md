# HVGN

## Introduction
This repository described in the paper "Combining Fundus Images and Fluorescein Angiography for Artery/Vein Classification Using the Hierarchical Vessel Graph Network" (https://link.springer.com/chapter/10.1007/978-3-030-59722-1_57)
![image](https://user-images.githubusercontent.com/64057617/156984481-49032ce7-ab33-4148-b259-fc7b6f11417d.png)
## Usage

### Installation
```
git clone snubhretina/HVGN
cd HVGN
```

* Download the pretrained Vessel extraction models form [here]. This model is trained DRIVE Database. Our model can't provide cause trained our SNUBH internel DB.
* Unzip and move the pretrained parameters to models/

### Run
```
python main.py --input_path="./data" --output_path="./res/" --seg_model_path = "./model/seg_model.pth" --gnn_model_path = "./model/seg_model.pth"
```
you must input model


## Citation
```
@inproceedings{noh2020combining,
  title={Combining fundus images and fluorescein angiography for artery/vein classification using the hierarchical vessel graph network},
  author={Noh, Kyoung Jin and Park, Sang Jun and Lee, Soochahn},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={595--605},
  year={2020},
  organization={Springer}
}
```
