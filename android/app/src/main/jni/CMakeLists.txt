project(makeup)

cmake_minimum_required(VERSION 3.4.1)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20211208-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.5.1-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

add_library(${PROJECT_NAME} SHARED makeup_jni.cpp parsing.cpp transfer.cpp blazeface.cpp)

target_link_libraries(${PROJECT_NAME} ncnn ${OpenCV_LIBS})
