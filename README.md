# YOLOV1-PYTORCH
## 1.Description  
This is YOLOV1 implemented by me using pytorch based on YOLOV1 paper [https://arxiv.org/abs/1506.02640](https://arxiv.org/abs/1506.02640).All code implementations are based on my own understanding of the paper, so there may be understanding deviations, please understand.
## 2.Train
1.Modify the configuration file conf.json,'common' is a common parameter for testing and training,'train' is a training related parameter, and'test' is a test related parameter.It should be noted that S represents the number of grid cells, which is 7 in the original paper, and B represents the number of bounding boxes predicted by each grid cell, which is 2 in the original paper.  
2.Make your own data set according to the VOC2012 data set format or use the VOC data set directly,if use your own dataset,you can use makeTxT.py to split your dataset.'voc_Annotations_dir' in 'train' represent the Annotations folder path in VOC2012 dataset;'voc_Main_dir' in 'train' represent the Main folder path in VOC2012 dataset;'voc_JPEGImages_dir' in 'train' represent the JPEGImages folder path in VOC2012 dataset.  
3.run 'python train.py'
## 3.Test
1.Modify the 'test' configuration in conf.json.  
2.run 'python test.py'
## 4.Result  

![](https://github.com/1991yuyang/YOLOV1-PYTORCH/blob/master/predict_result/2008_005534.jpg?raw=true)  
![](https://github.com/1991yuyang/YOLOV1-PYTORCH/blob/master/predict_result/2008_007538.jpg?raw=true)  
![](https://github.com/1991yuyang/YOLOV1-PYTORCH/blob/master/predict_result/2011_004694.jpg?raw=true)  
![](https://github.com/1991yuyang/YOLOV1-PYTORCH/blob/master/predict_result/2011_004694.jpg?raw=true)
