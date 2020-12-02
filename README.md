# CenterX

This repo is implemented based on [detectron2](https://github.com/facebookresearch/detectron2) and  [CenterNet](https://github.com/xingyizhou/CenterNet)

## What\'s new
- Support [imgaug](https://github.com/aleju/imgaug.git) data augmentation
- Support [swa](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/)
- Support **Knowledge Distill**, teacher-student, designed by myself
- Support other LR_SCHEDULER
- Support Optimizer [RangerLars](https://github.com/pabloppp/pytorch-tools.git), not convergence in COCO
- We provide some examples and scripts to convert centerX to Caffe, ONNX and TensorRT format in [projects/speedup](https://github.com/CPFLAME/centerX/tree/master/projects/speedup)
 
## What\'s comming 
- [️✔] Support simple inference 
- [✔] Support to caffe, onnx, tensorRT
- [ ] Support keypoints 

## Requirements

- Python >= 3.7
- PyTorch >= 1.5
- torchvision that matches the PyTorch installation.
- OpenCV
- pycocotools

```shell
pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

- GCC >= 4.9

```shell
gcc --version
```

- detectron2

```shell
pip install -U 'git+https://github.com/CPFLAME/detectron2.git'
```

- [pytorch tools](https://github.com/pabloppp/pytorch-tools.git)

```shell
pip install git+https://github.com/pabloppp/pytorch-tools -U
```
### Data prepare
the same as [detectron2](https://detectron2.readthedocs.io/tutorials/builtin_datasets.html)

### Training

modify your yamls in run.sh
```shell
sh run.sh
```

### Testing and Evaluation

modify your yamls in run.sh, 
add eval-only and MODEL.WEIGHTS in your setting
```shell
sh run.sh
```

## Performance

### coco

This repo use less training time to get a competitive performance compared to other versions

Backbone ResNet-50

| Code             | mAP  | epoch |
| ---------------- | ---- | ----- |
| centerX          | 33.2 |  70   |
| centerX          | 34.3 |  140  |
| centernet-better | 34.9 |  140  |

Backbone ResNet-18

centerX_KD means ResNet-50(33.2) as teacher, ResNet-18(27.9) as student, Knowledge Distill for 70 epoch in coco.

| Code             | mAP  | epoch |
| ---------------- | ---- | ----- |
| centerX          | 30.2 |  140  |
| centerX          | 27.9 |  70   |
| centerX_KD       | 31.0 |  70   |
| centernet-better | 29.8 |  140  |
| centernet        | 28.1 |  140  |

### crowd human
- optim: SGD
- lr: 0.02
- scheduler: WarmupMultiStepLR ,drop 0.1 in (50, 62) for 80 epoch; (90 ,120) for 140 epoch 
- train size: 512 max size
- test size: 512 max size
- batch size: 64
- woGT: KD only use teacher loss


| Backbone                 | mAP  | mAP50 | mAP75 | epoch | teacher | student_pretrain |
| ----------------         | ---- | ----- | ----- | ----- |  -----  | ------- |
| resdcn18                 | 31.2 | 56.6  | 30.8  |  80   |   -     |  -      |
| resdcn18_swa             | 31.1 | 56.6  | 30.4  |  80   |   -     |  -      |
| resdcn18_syncBN          | 31.3 | 56.6  | 30.7  |  80   |   -     |  -      |
| resdcn18_imgaug          | 29.6 | 54.7  | 28.9  |  80   |   -     |  -      |
| resdcn18_KD              | 34.5 | 60.2  | 34.3  |  80   | resdcn50| resdcn18|
| resdcn18_KD_woGT         | 33.0 | 58.3  | 32.7  |  80   | resdcn50| resdcn18|
| resdcn18_KD_woGT_scratch | 32.8 | 58.1  | 32.6  |  140  | resdcn50| imagenet|
| resdcn50                 | 35.1 | 61.2  | 35.3  |  80   |   -     |  -      |
 
## KD exp

### crowd human KD
Generalization performance for Knowledge Distill

| Backbone                 | crowd mAP  | coco_person mAP  | epoch | teacher | student_pretrain | train_set |
| ----------------         | ----       | --------------   | ----- |  -----  | -------          | -----     |
| resdcn50                 | **35.1**   |    35.7          |  80   |   -     |    -             | crowd     |
| resdcn18(baseline)       | 31.2       |    31.2          |  80   |   -     |    -             | crowd     |
| resdcn18_KD              | 34.5       |    34.9          |  80   | resdcn50| resdcn18         | crowd     |
| resdcn18_KD_woGT_scratch | 32.8       |    34.2          |  140  | resdcn50| imagenet         | crowd     |
| resdcn18_KD_woGT_scratch | 34.1       |  **36.3**        |  140  | resdcn50| imagenet         | crowd+coco|

### multi teacher KD

| Backbone                 |  mAP crowd     |  mAP coco_car  | epoch | teacher | student_pretrain | train_set |
| ----------------         | ----           | ----           | ----- |  -----  | -------          | -----     |
| 1.resdcn50               | 35.1           | -              |  80   |   -     |    -             | crowd     |
| 2.resdcn18               | 31.7           | -              |  70   |   -     |    -             | crowd     |
| 3.resdcn50               | -              | 31.6           |  70   |   -     |    -             | coco_car  |
| 4.resdcn18               | -              | 27.8           |  70   |   -     |    -             | coco_car  |
| resdcn18_KD_woGT_scratch | 31.6           | 29.4           |  140  |  1,3    | imagenet         | crowd+coco_car|

| Backbone                 |  mAP crowd_human |  mAP widerface | epoch | teacher | student_pretrain | train_set |
| ----------------         | ----             | -------------- | ----- |  -----  | -------          | -----     |
| 1.resdcn50               | 35.1             | -              |  80   |   -     |    -             | crowd     |
| 2.resdcn18               | 31.7             | -              |  70   |   -     |    -             | crowd     |
| 3.resdcn50               | -                | 32.9           |  70   |   -     |    -             | widerface |
| 4.resdcn18               | -                | 29.6           |  70   |   -     |    -             | widerface |
| 5.resdcn18_ignore_nolabel| 29.1             | 24.2           |  140   |   -     |    -             |crowd+wider|
| 6.resdcn18_pseudo_label  | 28.9             | 27.7           |  140   |   -     |    -             |crowd+wider|
| 7.resdcn18_KD_woGT_scratch| 31.3            | 32.1           |  140  |  1,3    | imagenet         |crowd+wider|

## License

centerX is released under the [Apache 2.0 license.](https://github.com/CPFLAME/centerX/blob/master/LICENSE)

## Acknowledgement

- [detectron2](https://github.com/facebookresearch/detectron2)
- [CenterNet](https://github.com/xingyizhou/CenterNet)
- [CenterNet-better](https://github.com/FateScript/CenterNet-better)
- [CenterNet-better-plus](https://github.com/lbin/CenterNet-better-plus.git)
- [FastReID](https://github.com/JDAI-CV/fast-reid)
