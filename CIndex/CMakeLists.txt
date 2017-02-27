cmake_minimum_required(VERSION 2.8)
PROJECT('CIndex')
#add_definitions(-std=c++11 -O3)
SET(CMAKE_BUILD_TYPE Release)
#SET(CMAKE_BUILD_TYPE Debug)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

AUX_SOURCE_DIRECTORY(. SRC_DIR)
#AUX_SOURCE_DIRECTORY(../src SRC_DIR_DARKNET)
INCLUDE_DIRECTORIES(/usr/local/cuda/include)
INCLUDE_DIRECTORIES(/usr/local/cudnn-v5/include)
LINK_DIRECTORIES(/usr/local/cuda/lib64)
LINK_DIRECTORIES(/usr/local/cudnn-v5/lib64)

get_filename_component(DARKNET_SO_PATH ${CMAKE_SOURCE_DIR} PATH)
message("libdarknet.so path : " ${DARKNET_SO_PATH})

add_definitions(-DGPU)
add_definitions(-DCUDNN)

# packages
#find_package( OpenCV REQUIRED )

#ADD_LIBRARY(DarknetPre SHARED ${SRC_DIR} ${SRC_DIR_DARKNET})
ADD_LIBRARY(DarknetPre SHARED ${SRC_DIR})

#TARGET_LINK_LIBRARIES(DarknetPre libflann.so  ${OpenCV_LIBS})
# link libdarknet.so 
TARGET_LINK_LIBRARIES(DarknetPre ${DARKNET_SO_PATH}/libdarknet.so)
set_target_properties(DarknetPre PROPERTIES PREFIX "")
