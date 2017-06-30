###### use  
1. **complie darknet as a share lib**  
make
2. **make darknet python export**  
cd CIndex; cmake CMakeLists.txt; make
3. **test darknet python export**  
cd ..; python test.py

###### tips  
* change test gpu id  
fix src/cuda.c line 1 
* other darknet python export  
[py_yolov2]()https://github.com/SidHard/py-yolo2.git
  
![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

#Darknet#
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).
