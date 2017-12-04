CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

project(cuda_compare_normal_optimize)

#folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


#openMP C++
FIND_PACKAGE(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -std=c++11")

#Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
if(NOT OpenCV_FOUND)
    message(ERROR " OpenCV not found!")
endif(NOT OpenCV_FOUND)

#CUDA
find_package(CUDA)
if (NOT CUDA_FOUND)
	message("CUDA not found!")
endif(NOT CUDA_FOUND)


CUDA_ADD_EXECUTABLE(normal normal.cu timer.hpp timer.cpp)
TARGET_LINK_LIBRARIES(normal ${OpenCV_LIBS})

CUDA_ADD_EXECUTABLE(normal_openMP normal_openMP.cu timer.hpp timer.cpp)
TARGET_LINK_LIBRARIES(normal_openMP ${OpenCV_LIBS})

CUDA_ADD_EXECUTABLE(cuda cuda.cu timer.hpp timer.cpp)
TARGET_LINK_LIBRARIES(cuda ${OpenCV_LIBS})

CUDA_ADD_EXECUTABLE(cuda_constant cuda_constant.cu timer.hpp timer.cpp)
TARGET_LINK_LIBRARIES(cuda_constant ${OpenCV_LIBS})

CUDA_ADD_EXECUTABLE(cuda_texture cuda_texture.cu timer.hpp timer.cpp)
TARGET_LINK_LIBRARIES(cuda_texture ${OpenCV_LIBS})

CUDA_ADD_EXECUTABLE(cuda_texture_2d cuda_texture_2d.cu timer.hpp timer.cpp)
TARGET_LINK_LIBRARIES(cuda_texture_2d ${OpenCV_LIBS})

CUDA_ADD_EXECUTABLE(cuda_texture_constant cuda_texture_constant.cu timer.hpp timer.cpp)
TARGET_LINK_LIBRARIES(cuda_texture_constant ${OpenCV_LIBS})

