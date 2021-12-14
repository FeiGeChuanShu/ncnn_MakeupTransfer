// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

// ncnn
#include "net.h"
#include "benchmark.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "parsing.h"
#include "transfer.h"
#include "blazeface.h"
static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static Transfer makeup_transfer;
static Parsing face_parsing;
static BlazeFace face_detect;
extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "makeup", "JNI_OnLoad");

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "makeup", "JNI_OnUnload");

    ncnn::destroy_gpu_instance();
}

// public native boolean Init(AssetManager mgr);
JNIEXPORT jboolean JNICALL Java_com_tencent_makeup_Makeup_Init(JNIEnv* env, jobject thiz, jobject assetManager)
{
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    if(face_parsing.load(mgr) < 0)
        return JNI_FALSE;
    if(makeup_transfer.load(mgr) < 0)
        return JNI_FALSE;
    if(face_detect.load(mgr) < 0)
        return JNI_FALSE;

    return JNI_TRUE;
}

// public native Bitmap StyleTransfer(Bitmap bitmap, int style_type, boolean use_gpu);
JNIEXPORT jboolean JNICALL Java_com_tencent_makeup_Makeup_Process(JNIEnv* env, jobject thiz, jobject target_bitmap, jobject reference_bitmap,jint style_type, jboolean use_gpu)
{
    if (style_type < 0 || style_type >= 5)
        return JNI_FALSE;

    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0)
        return JNI_FALSE;

    double start_time = ncnn::get_current_time();

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, target_bitmap, &info);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return JNI_FALSE;
    AndroidBitmap_getInfo(env, reference_bitmap, &info);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return JNI_FALSE;

    ncnn::Mat target_img = ncnn::Mat::from_android_bitmap(env, target_bitmap, ncnn::Mat::PIXEL_RGB);
    ncnn::Mat reference_img = ncnn::Mat::from_android_bitmap(env, reference_bitmap, ncnn::Mat::PIXEL_RGB);

    cv::Mat non_makeup_img = cv::Mat::zeros(target_img.h,target_img.w,CV_8UC3);
    target_img.to_pixels(non_makeup_img.data, ncnn::Mat::PIXEL_RGB);
    cv::Mat makeup_img = cv::Mat::zeros(reference_img.h,reference_img.w,CV_8UC3);
    reference_img.to_pixels(makeup_img.data, ncnn::Mat::PIXEL_RGB);

    std::vector<FaceObject> faceobjects;
    face_detect.detect(non_makeup_img,faceobjects);
    if(faceobjects.size() == 0)
        return JNI_FALSE;

    cv::Mat non_makeup_img_parsing;
    face_parsing.parsing(non_makeup_img(faceobjects[0].rect).clone(), non_makeup_img_parsing, use_gpu);
    
    cv::Mat makeup_img_parsing;
    face_parsing.parsing(makeup_img, makeup_img_parsing, use_gpu);

    cv::Mat makeup_result;
    makeup_transfer.transfer(non_makeup_img(faceobjects[0].rect).clone(), makeup_img, non_makeup_img_parsing, makeup_img_parsing, makeup_result, use_gpu);

    ncnn::Mat blengImg_ncnn = ncnn::Mat::from_pixels(makeup_result.data,ncnn::Mat::PIXEL_RGB,makeup_result.cols,makeup_result.rows);

    // ncnn to bitmap
    blengImg_ncnn.to_android_bitmap(env, target_bitmap, ncnn::Mat::PIXEL_RGB);

    double elasped = ncnn::get_current_time() - start_time;
    __android_log_print(ANDROID_LOG_DEBUG, "makeup", "%.2fms", elasped);

    return JNI_TRUE;
}

}
