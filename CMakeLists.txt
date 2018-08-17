project(GMS)

cmake_minimum_required(VERSION 2.6)
aux_source_directory(${PROJECT_SOURCE_DIR}/src DIR_SRCS)
add_definitions("-Wall -g -std=c++11")
option(USE_GPU "select to use gpu" OFF)
message("USE_GPU is ${USE_GPU}")

if(USE_GPU)
    add_definitions(-DUSE_GPU)
	set(LIBS ${LIBS} gms_gpu)
    message("select to use library name: ${LIBS}")
else()
	set(LIBS ${LIBS} gms)
    message("select to use library name: ${LIBS}")
endif(USE_GPU)

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
	
add_library(${LIBS} SHARED ${DIR_SRCS})
target_link_libraries(
	${LIBS}
	opencv_core
	opencv_highgui
	opencv_imgproc
	opencv_imgcodecs
	opencv_videoio
	opencv_xfeatures2d
	opencv_features2d
	opencv_xobjdetect
	opencv_objdetect
	opencv_cudacodec
	opencv_cudafeatures2d
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
	opencv_cudacodec
	opencv_cudafeatures2d
	opencv_flann
	)
