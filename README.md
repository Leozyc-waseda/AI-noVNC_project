# #  Yolov3_train_your_own_dataset
## This readme will teach you how to train your own YOLO with your own dataset



## Steps

- Get darknet.
- Label your own dataset(by using labelImg).
- Organize your own dataset.
- Modify the configuration file.
- Train your own dataset.
- Test the trained network model.
- Performance statistics.



## Get darknet

- clone darknet
  ```python
  $ git clone https://github.com/pjreddie/darknet
  ```
- build
  ```python
  $ cd darknet
  $ make
  ```
- download the pre-trained weight file
  ```python
  $ wget https://pjreddie.com/media/files/yolov3.weights
  ```
- test
  ```python
  $ ./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
  ```
    The predictions.jpg under the directory darknet is the generated prediction result image file
- Compiling With CUDA and OPENCV
  - CUDA installation and OpenCV installation (make your self)
  - Modify darknet's Makefile
  ```python
    GPU=1
    CUDNN=1
    OPENCV=1
    $ make clean
    $ make
  ```
- Test the GPU version of yolo
   ```python
    $ ./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
  ```
  Prediction time reduced to about 0.0xx seconds

## Label your own dataset
- Get labelImg
  ```python
  $ git clone https://github.com/tzutalin/labelImg.git
  $ cd labelImg
  $ python labelImg.py
  ```
- Modify the file labelImg/data/predefined_classes.txt
  ```python
  your own class label
  for example: 
  dog
  cat
  coffee
  ```
- Image annotation with labelImg
  ```python
  $ cd labelImg
  $ python labelImg.py
  ```
  Remember to select the format as yolo. If it is pascal voc format using ``` Yolov3_train_your_own_dataset/train_yolo/gen_files.py``` to transfer.


## Organize your own dataset
- Make your folder like below
  ```python
  darknet / VOCdevkit / VOC2007 / Annotations
                                                            / JPEGImages
  ```
- Generate train and test files
  ```python
  $ python ~/train_yolo/gen_files.py
  ```
  In the ```VOCdevkit / VOC2007 ```directory, you can see that the folder labels is generated, and two files ```2007_train.txt ```and ```2007_test.txt``` are generated under darknet. ```2007_train.txt``` and ```2007_test.txt``` give lists of training image files and test image files, respectively, with the path and filename of each image. In addition, two files ```test.txt``` and ```train.txt``` are generated in the ```VOCdevkit/VOC2007/ImageSets/Main``` directory, which give lists of training image files and test image files respectively, but only contain the file name of each image (not with path and extension).

## Modify the configuration file
- Create a new ```data/voc.names``` file, you can copy ```data/voc.names``` and modify it according to your own situation; you can rename it such as: ```data/voc-tired.names```
- To create a new ```cfg/voc.data file```, you can copy ```cfg/voc.data``` and modify it according to your own situation; you can rename it such as: ```cfg/voc-tired.data```
- To create a new ```cfg/yolov3-voc.cfg```, you can copy ```cfg/yolov3-voc.cfg``` and modify it according to your own situation; you can rename ```cfg/yolov3-tired.cfg```: In the ```cfg/yolov3-voc.cfg``` file, three The parameters of the yolo layer and the conv layers in front of them need to be modified: The three yolo layers must be changed: the class in the yolo layer is the number of categories, and the filters in the conv layer before each yolo layer = ```(category + 5) * 3 ```For example :
    - yolo layer classes=1, conv layer filters=18
    - yolo layer classes=2, conv layer filters=21
    - yolo layer classes=4, conv layer filters=27
#### Train your own dataset
- Download the weights file in the darknet directory:
  ```python
  $ wget https://pjreddie.com/media/files/darknet53.conv.74
  ```
- Train
  ```python
  $ ./darknet  detector train cfg/voc-tired.data cfg/yolov3-tired.cfg darknet53.conv.74
    #To store training logs, execute
  $ ./darknet detector train cfg/voc-ball.data cfg/yolov3-voc-ball.cfg darknet53.conv.74 2>1 | tee visualization/train_yolov3_ball.log
  ```
  Output information during training: Region 106 Avg IOU: 0.794182, Class: 0.999382, Obj: 0.966953, No Obj:
0.000240, .5R: 1.000000, .75R: 0.750000, count: 4
    - Region 106 : The index of the network layer is 106
    - Region Avg IOU: 0.794182: Indicates the average IOU of the pictures in the current l.batch (batch /= subdivs )
    - Class: 0.0.999382: Label the correct rate of target classification, expect the value to be close to 1.
    - Obj: 0.966953: The average target confidence of the detected target, the closer to 1, the better.
    - No Obj: 0.000793: Average objectivity score for detected objects.
    - .5R: 1.0: The ratio of positive samples detected by the model (iou>0.5) to actual positive samples.
    - .75R: 0.75 The ratio of positive samples detected by the model (iou>0.75) to actual positive samples.
    - count: The value after 4:count is the number of pictures that contain positive samples in the pictures in the current l.batch (batch /= subdivs ).
- Training log file analysis:
  ```python
  $ cd visualization
  $ python extract_log.py
  #Get two files: train_log_loss.txt, train_log_iou.txt Change the value of lines in it
  $ python train_loss_visualization.py
  $ python train_iou_visualization.py
  ```
  get avg_loss.png and Region Avg IOU.png

- Training tips:
  - batch=64
  - subdivisions=16 (8 available when memory is large)
  - Set max_batches to (classes*2000); but at least 4000. For example, if training 3 target classes, max_batches=6000
  - Change steps to 80% and 90% of max_batches; for example, steps=4800, 5400 can increase the value of height and width to increase the network resolution, but it must be a multiple of 32. cfg-file (height=608, width=608 or any value multiple of 32) . This helps to improve detection accuracy.
  
#### Test the trained network model
- Test pictures:
  ```python
  $ ./darknet detector test cfg/voc-tired.data cfg/yolov3-voc-tired.cfg backup/yolov3-voc-final.weights testfiles/img1.jpg
  ```
- Test videos:
  ```python
  $ ./darknet detector demo cfg/voc-tired.data cfg/yolov3-voc-tired.cfg backup/yolov3-voc-final.weights testfiles/xxx.mp4
  ```

