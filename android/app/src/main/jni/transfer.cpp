// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "transfer.h"
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

int Transfer::load(AAssetManager* mgr,  bool use_gpu)
{
    transfer_net.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    transfer_net.opt = ncnn::Option();

#if NCNN_VULKAN
    transfer_net.opt.use_vulkan_compute = use_gpu;
#endif

    transfer_net.opt.num_threads = ncnn::get_big_cpu_count();

    int ret_param = transfer_net.load_param(mgr, "makeup.param");
    int ret_bin = transfer_net.load_model(mgr, "makeup.bin");
    if (ret_param != 0 || ret_bin != 0)
    {
        return -1;
    }
    target_size = 256;

    return 0;
}
int Transfer::transfer(const cv::Mat& non_makeup_img, const cv::Mat& makeup_img,
                    const cv::Mat& non_makeup_parse_img, const cv::Mat& makeup_parse_img,
                    cv::Mat &makeup_result, bool use_gpu)
{
    const int target_width = target_size;
    const int target_height = target_size;

    ncnn::Mat non_makeup_parse = ncnn::Mat(target_width, target_height, 18);
    ncnn::Mat makeup_parse = ncnn::Mat(target_width, target_height, 18);
    non_makeup_parse.fill(0.f);
    makeup_parse.fill(0.f);

    cv::Mat non_makeup_parse_img_resize, makeup_parse_img_img_resize;
    cv::resize(non_makeup_parse_img, non_makeup_parse_img_resize, cv::Size(256, 256), 0, 0, cv::INTER_NEAREST);
    cv::resize(makeup_parse_img, makeup_parse_img_img_resize, cv::Size(256, 256), 0, 0, cv::INTER_NEAREST);
    for (int i = 0; i < non_makeup_parse_img_resize.rows; i++)
    {
        for (int j = 0; j < non_makeup_parse_img_resize.cols; j++)
        {
            non_makeup_parse.channel(non_makeup_parse_img_resize.at<uchar>(i, j))[i * target_width + j] = 1.0;
        }
    }
    for (int i = 0; i < makeup_parse_img_img_resize.rows; i++)
    {
        for (int j = 0; j < makeup_parse_img_img_resize.cols; j++)
        {
            makeup_parse.channel(makeup_parse_img_img_resize.at<uchar>(i, j))[i * target_width + j] = 1.0;
        }
    }

    ncnn::Extractor ex = transfer_net.create_extractor();
    ncnn::Mat non_makeup = ncnn::Mat::from_pixels_resize(non_makeup_img.data, ncnn::Mat::PIXEL_RGB, non_makeup_img.cols, non_makeup_img.rows, target_width, target_height);
    ncnn::Mat makeup = ncnn::Mat::from_pixels_resize(makeup_img.data, ncnn::Mat::PIXEL_RGB, makeup_img.cols, makeup_img.rows, target_width, target_height);

    non_makeup.substract_mean_normalize(mean_vals, norm_vals);
    makeup.substract_mean_normalize(mean_vals, norm_vals);

    ex.set_vulkan_compute(use_gpu);
    
    ex.input("non_makeup", non_makeup);
    ex.input("makeup", makeup);
    ex.input("non_makeup_parse", non_makeup_parse);
    ex.input("makeup_parse", makeup_parse);
    ncnn::Mat out_330;
    ex.extract("330", out_330);
    ncnn::Mat out_339;
    ex.extract("339", out_339);

    for (int c = 0; c < out_330.c; c++)
    {
        float* out_330_data = out_330.channel(c);
        for (int i = 0; i < out_330.h; i++)
        {
            for (int j = 0; j < out_330.w; j++)
            {
                out_330_data[i * out_330.w + j] = out_330_data[i * out_330.w + j] / out_339[j];
            }
        }
    }

    ncnn::Mat out_336;
    ex.extract("336", out_336);
    ncnn::Mat out_343;
    ex.extract("343", out_343);

    for (int c = 0; c < out_336.c; c++)
    {
        float* out_336_data = out_336.channel(c);
        for (int i = 0; i < out_336.h; i++)
        {
            for (int j = 0; j < out_336.w; j++)
            {
                out_336_data[i * out_336.w + j] = out_336_data[i * out_336.w + j] / out_343[j];
            }
        }
    }

    ex.input("340", out_330);
    ex.input("344", out_336);
    ncnn::Mat out;
    ex.extract("out", out);

    const float mean_vals1[3] = { -1.0, -1.0, -1.0 };
    const float norm_vals1[3] = { 1 / 2.0, 1 / 2.0, 1 / 2.0 };
    out.substract_mean_normalize(mean_vals1, norm_vals1);

    cv::Mat makeup_result_img_32F = cv::Mat::zeros(cv::Size(256,256),CV_32FC3);

    for (int i = 0; i < target_height; i++)
    {
        for (int j = 0; j < target_width; j++)
        {
            makeup_result_img_32F.at<cv::Vec3f>(i, j)[0] = out.channel(0)[i * target_width + j] * 255;
            makeup_result_img_32F.at<cv::Vec3f>(i, j)[1] = out.channel(1)[i * target_width + j] * 255;
            makeup_result_img_32F.at<cv::Vec3f>(i, j)[2] = out.channel(2)[i * target_width + j] * 255;
        }
    }

    cv::Mat makeup_result_img_8U;
    makeup_result_img_32F.convertTo(makeup_result_img_8U, CV_8UC3, 1.0, 0);
    cv::resize(makeup_result_img_8U, makeup_result_img_8U, non_makeup_img.size(), 0, 0, cv::INTER_LINEAR);
    makeup_result_img_8U.copyTo(makeup_result);


    return 0;
}
