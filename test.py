import sys
import os

# test darknet so
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/CIndex/bin/')
import DarknetPre 

print DarknetPre 
print DarknetPre.predict

# predict
net = DarknetPre.init_model('cfg/coco.data', 'cfg/yolo.cfg', 'yolo.weights', 4)
print 'net == ', net

for i in range(1):
    result = DarknetPre.predict(
         net
        , 'data/dog.jpg'
        , .24
        , .5
        )
    print result
