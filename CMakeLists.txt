project(GMS)

cmake_minimum_required(VERSION 2.6)
aux_source_directory(${PROJECT_SOURCE_DIR}/src DIR_SRCS)
add_definitions("-Wall -g -std=c++11")
add_definitions(-DUSE_GPU=OFF)

include_directories(
	/usr/include/
	/usr/local/include
	/usr/local/cuda/include
	)

	
link_directories(
	/usr/lib
	/usr/local/lib
	/usr/lib/x86_64-linux-gnu
	/usr/local/cuda/lib64
	)
	
add_library(gms SHARED ${DIR_SRCS})
target_link_libraries(
	gms
	opencv_core
	opencv_highgui
	opencv_imgproc
	opencv_imgcodecs
	opencv_videoio
	opencv_xfeatures2d
	opencv_features2d
	opencv_xobjdetect
	opencv_objdetect
	opencv_flann
	)
	
add_executable(GMS ${DIR_SRCS})
target_link_libraries(
	GMS
	opencv_core
	opencv_highgui
	opencv_imgproc
	opencv_imgcodecs
	opencv_videoio
	opencv_xfeatures2d
	opencv_features2d
	opencv_xobjdetect
	opencv_objdetect
	opencv_flann
	)
