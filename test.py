import sys
import os

# test darknet so
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/CIndex/bin/')
import DarknetPre 

print DarknetPre 
print DarknetPre.predict

# predict
cfg_root = '/world/data-c9/yolo_online_cfg/'
#net = DarknetPre.init_model('cfg/coco.data', 'cfg/yolo.cfg', 'yolo.weights', 4)
net = DarknetPre.init_model(cfg_root + 'cfg/coco.data', cfg_root + 'cfg/yolo.cfg', cfg_root + 'weights/yolo.weights', 4)
print 'net == ', net
for i in range(1):
    result = DarknetPre.predict(
         net
        , 'data/dog.jpg'
        , .24
        , .5
        )
    print result

import cv2
im_data = cv2.imread('data/dog.jpg', cv2.IMREAD_COLOR)
#im_data = cv2.resize(im_data, (416, 416))
print type(im_data)
res = DarknetPre.detect(net, im_data, .24, .5)
print res
