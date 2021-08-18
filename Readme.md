
Model Comparision
----------------

| Experiments | Parameters | Best Training Acc | Best Training Loss | Best Test Acc  | Best Test Loss  | Analysis |
| :-------: | :---------: |:-----------: | :------: | :-------: | :-------: | :-------: |
| [Model4_best_modelLR](https://github.com/vivek-a81/EVA6/blob/main/Session5/Model4_best_modelLR.ipynb) |  7.4k | 99.24 | 0.00553 |99.56% | 0.0168 |  • We tried different learning rate schedulers like ReduceLRPlatue which did not work because on platue decreses the learning rate when loss is incresing in a particular epochs but this did not happen in out problem. StepLR also did not work because it is hard to predict at which point the learning rate should be reduced <br> • So, we have used OneCycleLR which updates learning rate throughout the epoch with every single batch is passed (Learning rate is updated in a particular range) |
| [Model3_augmentation](https://github.com/vivek-a81/EVA6/blob/main/Session5/Model3_augmentation.ipynb) |  7.4k |  99.00 | 0.01188 | 99.34 | 0.0213 | There were some examples in testing data on which model was not able to give correct prediction, adding data augmentation helps to modifies the training samples, which increse the performance of model while training, we can see it as the jump in training accuracy and also the model is able to predict the testing data better compare with last model. |
| [Model2_bn_dropout](https://github.com/vivek-a81/EVA6/blob/main/Session5/Model2_bn_dropout.ipynb) | 7.4k | 98.58 | 0.01337 | 99.28 | 0.0230 | The model is pushed future by applying the batchnormalization. Dropout made the training harder by dropping the neurons |
| [Model1_Basic_Skeleton](https://github.com/vivek-a81/EVA6/blob/main/Session5/Model1_Basic_Skeleton.ipynb) | 7.3k | 98.58 | 0.01337 | 98.66 | 0.0464 | Design model architeture of less than 8000 parameters The accuracy was very well with such less number of parameters and the training accuracy batch normalization and dropout can push the performance |



# Best Model Architecture
  - [to notebbok](https://github.com/vivek-a81/EVA6/blob/main/Session5/Model4_best_modelLR.ipynb)
  
![alt](Images/SESS5.png)

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
       BatchNorm2d-2            [-1, 8, 26, 26]              16
              ReLU-3            [-1, 8, 26, 26]               0
           Dropout-4            [-1, 8, 26, 26]               0
            Conv2d-5           [-1, 16, 24, 24]           1,152
       BatchNorm2d-6           [-1, 16, 24, 24]              32
              ReLU-7           [-1, 16, 24, 24]               0
           Dropout-8           [-1, 16, 24, 24]               0
         MaxPool2d-9           [-1, 16, 12, 12]               0
           Conv2d-10            [-1, 8, 12, 12]             128
      BatchNorm2d-11            [-1, 8, 12, 12]              16
             ReLU-12            [-1, 8, 12, 12]               0
           Conv2d-13           [-1, 12, 10, 10]             864
      BatchNorm2d-14           [-1, 12, 10, 10]              24
             ReLU-15           [-1, 12, 10, 10]               0
          Dropout-16           [-1, 12, 10, 10]               0
           Conv2d-17             [-1, 16, 8, 8]           1,728
      BatchNorm2d-18             [-1, 16, 8, 8]              32
             ReLU-19             [-1, 16, 8, 8]               0
          Dropout-20             [-1, 16, 8, 8]               0
           Conv2d-21             [-1, 20, 6, 6]           2,880
      BatchNorm2d-22             [-1, 20, 6, 6]              40
             ReLU-23             [-1, 20, 6, 6]               0
          Dropout-24             [-1, 20, 6, 6]               0
        AvgPool2d-25             [-1, 20, 1, 1]               0
           Conv2d-26             [-1, 16, 1, 1]             320
      BatchNorm2d-27             [-1, 16, 1, 1]              32
             ReLU-28             [-1, 16, 1, 1]               0
          Dropout-29             [-1, 16, 1, 1]               0
           Conv2d-30             [-1, 10, 1, 1]             160
================================================================
Total params: 7,496
Trainable params: 7,496
Non-trainable params: 0
----------------------------------------------------------------
```
![](Images/rff.png)

| OPERATION |	N<sub>in</sub> |	N<sub>out</sub> |	CH<sub>in</sub> |	CH<sub>out</sub> |	Padding	| Kernel |	Stride	| j<sub>in</sub> |	j<sub>out</sub>	| r<sub>in</sub> |	r<sub>out</sub> |
| :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| **CONVLUTION** |	28 | 26	| 1	 | 8	| 0	| 3	| 1 | 1	| 1 |	1  |  3  |
| **CONVLUTION** |	26 | 24	| 8	 | 16 |	0	| 3	| 1	| 1 |	1 |	3  |	5  |
| **MaxPool**    |	24 | 12 |	16 | 16	| 0 |	2	| 2	| 1 |	2 | 5  |	6  |
| **CONVLUTION** |	12 | 12 | 16 |	8	| 0 |	1 |	1	| 2	| 2 |	6  |	6  |
| **CONVLUTION** |	12 | 10 |	8	 | 12	| 0 | 3 |	1	| 2	| 2 |	6  |	10 |
| **CONVLUTION** |	10 | 8  |	12 | 16 | 0 | 3 |	1	| 2	| 2	| 10 |	14 |
| **CONVLUTION** |	8	 | 6  | 16 | 20	| 0	| 3	| 1	| 2 |	2	| 14 |  18 |
| **GAP**        |  6  | 1  |	20 | 20	| 0	| 6	| 1 |	2 |	2 |	18 |	28 |
| **CONVLUTION** |	1	 | 1  |	20 | 16	| 0	| 1	| 1	| 2	| 2	| 28 |  28 |
| **CONVLUTION** |	1	 | 1  |	16 | 10	| 0	| 1 |	1	| 2	| 2	| 28 |	28 |


Learning Curve
------------

![alt](Images/loss.png)

Logs
-----------

```
  0%|          | 0/469 [00:00<?, ?it/s]
  EPOCH: 1
  Batch_id=468 Loss=0.27802 Accuracy=70.19: 100%|██████████| 469/469 [00:36<00:00, 12.89it/s]
    0%|          | 0/469 [00:00<?, ?it/s]

  Test set: Average loss: 0.2105, Accuracy: 9654/10000 (96.54%)

  EPOCH: 2
  Batch_id=468 Loss=0.09033 Accuracy=96.38: 100%|██████████| 469/469 [00:37<00:00, 12.68it/s]
    0%|          | 0/469 [00:00<?, ?it/s]

  Test set: Average loss: 0.0912, Accuracy: 9739/10000 (97.39%)

  EPOCH: 3
  Batch_id=468 Loss=0.03847 Accuracy=97.50: 100%|██████████| 469/469 [00:36<00:00, 12.68it/s]
    0%|          | 0/469 [00:00<?, ?it/s]

  Test set: Average loss: 0.0425, Accuracy: 9879/10000 (98.79%)

  EPOCH: 4
  Batch_id=468 Loss=0.02709 Accuracy=98.01: 100%|██████████| 469/469 [00:36<00:00, 12.72it/s]
    0%|          | 0/469 [00:00<?, ?it/s]

  Test set: Average loss: 0.0345, Accuracy: 9896/10000 (98.96%)

  EPOCH: 5
  Batch_id=468 Loss=0.03753 Accuracy=98.19: 100%|██████████| 469/469 [00:37<00:00, 12.50it/s]
    0%|          | 0/469 [00:00<?, ?it/s]

  Test set: Average loss: 0.0523, Accuracy: 9840/10000 (98.40%)

  EPOCH: 6
  Batch_id=468 Loss=0.04695 Accuracy=98.43: 100%|██████████| 469/469 [00:37<00:00, 12.44it/s]
    0%|          | 0/469 [00:00<?, ?it/s]

  Test set: Average loss: 0.0282, Accuracy: 9911/10000 (99.11%)

  EPOCH: 7
  Batch_id=468 Loss=0.01910 Accuracy=98.56: 100%|██████████| 469/469 [00:36<00:00, 12.71it/s]
    0%|          | 0/469 [00:00<?, ?it/s]

  Test set: Average loss: 0.0262, Accuracy: 9912/10000 (99.12%)

  EPOCH: 8
  Batch_id=468 Loss=0.00553 Accuracy=98.66: 100%|██████████| 469/469 [00:37<00:00, 12.65it/s]
    0%|          | 0/469 [00:00<?, ?it/s]

  Test set: Average loss: 0.0280, Accuracy: 9901/10000 (99.01%)

  EPOCH: 9
  Batch_id=468 Loss=0.02969 Accuracy=98.75: 100%|██████████| 469/469 [00:36<00:00, 12.71it/s]
    0%|          | 0/469 [00:00<?, ?it/s]

  Test set: Average loss: 0.0238, Accuracy: 9925/10000 (99.25%)

  EPOCH: 10
  Batch_id=468 Loss=0.01127 Accuracy=98.89: 100%|██████████| 469/469 [00:37<00:00, 12.63it/s]
    0%|          | 0/469 [00:00<?, ?it/s]

  Test set: Average loss: 0.0214, Accuracy: 9936/10000 (99.36%)

  EPOCH: 11
  Batch_id=468 Loss=0.03408 Accuracy=98.95: 100%|██████████| 469/469 [00:37<00:00, 12.58it/s]
    0%|          | 0/469 [00:00<?, ?it/s]

  Test set: Average loss: 0.0199, Accuracy: 9944/10000 (99.44%)

  EPOCH: 12
  Batch_id=468 Loss=0.00629 Accuracy=99.00: 100%|██████████| 469/469 [00:37<00:00, 12.53it/s]
    0%|          | 0/469 [00:00<?, ?it/s]

  Test set: Average loss: 0.0172, Accuracy: 9945/10000 (99.45%)

  EPOCH: 13
  Batch_id=468 Loss=0.02111 Accuracy=99.11: 100%|██████████| 469/469 [00:37<00:00, 12.51it/s]
    0%|          | 0/469 [00:00<?, ?it/s]

  Test set: Average loss: 0.0176, Accuracy: 9948/10000 (99.48%)

  EPOCH: 14
  Batch_id=468 Loss=0.02686 Accuracy=99.18: 100%|██████████| 469/469 [00:37<00:00, 12.51it/s]
    0%|          | 0/469 [00:00<?, ?it/s]

  Test set: Average loss: 0.0171, Accuracy: 9949/10000 (99.49%)

  EPOCH: 15
  Batch_id=468 Loss=0.02282 Accuracy=99.24: 100%|██████████| 469/469 [00:37<00:00, 12.47it/s]

  Test set: Average loss: 0.0168, Accuracy: 9956/10000 (99.56%)

```

Evaluation of Final Model 
-----------

![](Images/prediction.png)

