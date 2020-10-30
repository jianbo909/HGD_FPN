# HGD_FPN for object detection

##Introduction 

##Get Started
The implementation of HGD-FPN and the configs for training are in ```HGD_FPN``` directory.
You need to install mmdetection (version1.1.0 with mmcv 0.4.3).  
The files in ```HGD_FPN``` directory have same folder organization as mmdetecion.
More guidance can be found from [mmdeteion](https://github.com/open-mmlab/mmdetection).

## Models
The results on COCO 2017 val is shown in the below table.

| Method | Backbone | Add modules  | Lr schd | box AP | Download |
| :----: | :------: | :-------:  | :-----: | :----: | :------: |
| Faster RCNN | R-50-FPN | HGD-FPN |  1x  | 40.0| [model]()  |
| RetinaNet | R-50-FPN | HGD-FPN |  1x  | 37.8| [model]()  |
| FreeAnchor | R-50-FPN | HGD-FPN |  1x  | 40.8| [model]()  |

## Citations
Please cite our paper in your publications if it helps your research:
```
@inproceedings{wang2020SEPC,
    title     =  {A Holistically-Guided Decoder for Deep Representation Learning with Applications to Semantic Segmentation and Object Detection},
    author    =  {Jianbo, Liu and Sijie, Ren and Yuanjie, Zheng and Xiaogang, Wang and Hongsheng, Li},
    booktitle =  {Arxiv},
    year      =  {2020}
}
```

