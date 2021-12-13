#include "net.h"
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <stdlib.h>
#include <float.h>
#include <stdio.h>
#include <vector>

static int face_parsing(const ncnn::Net& net, const cv::Mat& in, cv::Mat& out)
{
    const int target_width = 512;
    const int target_height = 512;
    const int num_class = 19;

    ncnn::Extractor ex = net.create_extractor();
    const float mean_vals[3] = { 123.675f, 116.28f,  103.53f };
    const float norm_vals[3] = { 0.017058f, 0.017439f, 0.017361f };
    ncnn::Mat ncnn_in = ncnn::Mat::from_pixels_resize(in.data, ncnn::Mat::PIXEL_BGR2RGB, in.cols, in.rows, target_width, target_height);

    ncnn_in.substract_mean_normalize(mean_vals, norm_vals);

    ex.input("input", ncnn_in);
    ncnn::Mat output;
    ex.extract("output", output);

    out = cv::Mat::zeros(cv::Size(512, 512), CV_8UC1);
    float* output_data = (float*)output.data;

    int out_h = out.rows;
    int out_w = out.cols;
    for (int i = 0; i < out_h; i++)
    {
        for (int j = 0; j < out_w; j++)
        {
            int maxk = 0;
            float tmp = output_data[0 * out_w * out_h + i * out_w + j];
            for (int k = 0; k < num_class; k++)
            {
                if (tmp < output_data[k * out_w * out_h + i * out_w + j])
                {
                    tmp = output_data[k * out_w * out_h + i * out_w + j];
                    maxk = k;
                }
            }
            out.at<uchar>(i,j) = maxk;
        }
    }

    return 0;
}

static int makeup_transfer(const ncnn::Net& net,  const cv::Mat& non_makeup_img,const cv::Mat& makeup_img,
    const cv::Mat& non_makeup_parse_img,const cv::Mat& makeup_parse_img)
{
    const int target_width = 256;
    const int target_height = 256;

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

    ncnn::Extractor ex = net.create_extractor();
    const float mean_vals[3] = {127.5, 127.5, 127.5 };
    const float norm_vals[3] = {1 / 127.5, 1 / 127.5, 1 / 127.5 };
    ncnn::Mat non_makeup = ncnn::Mat::from_pixels_resize(non_makeup_img.data, ncnn::Mat::PIXEL_BGR2RGB, non_makeup_img.cols, non_makeup_img.rows, target_width, target_height);
    ncnn::Mat makeup = ncnn::Mat::from_pixels_resize(makeup_img.data, ncnn::Mat::PIXEL_BGR2RGB, makeup_img.cols, makeup_img.rows, target_width, target_height);

    non_makeup.substract_mean_normalize(mean_vals, norm_vals);
    makeup.substract_mean_normalize(mean_vals, norm_vals);

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
            makeup_result_img_32F.at<cv::Vec3f>(i, j)[2] = out.channel(0)[i * target_width + j] * 255;
            makeup_result_img_32F.at<cv::Vec3f>(i, j)[1] = out.channel(1)[i * target_width + j] * 255;
            makeup_result_img_32F.at<cv::Vec3f>(i, j)[0] = out.channel(2)[i * target_width + j] * 255;
        }
    }
    

    cv::Mat makeup_result_img_8U;
    makeup_result_img_32F.convertTo(makeup_result_img_8U, CV_8UC3, 1.0, 0);
    cv::resize(makeup_result_img_8U, makeup_result_img_8U, non_makeup_img.size(), 0, 0, cv::INTER_LINEAR);
    cv::imshow("non_makeup", non_makeup_img);
    cv::imshow("reference", makeup_img);
    cv::imshow("target", makeup_result_img_8U);
    cv::waitKey();
    return 0;
}

int process(const cv::Mat& target_img, const cv::Mat& reference_img)
{
    ncnn::Net parsing_net;
    parsing_net.opt.num_threads = 4;
    int ret_param = parsing_net.load_param("./models/face_parsing.param");
    int ret_bin = parsing_net.load_model("./models/face_parsing.bin");
    if (ret_param != 0 || ret_bin != 0) {
        fprintf(stderr,"parsing_net load param(%d), model(%d)\n", ret_param, ret_bin);
        return -1;
    }

    ncnn::Net makeup_net;
    makeup_net.opt.num_threads = 4;
    ret_param = makeup_net.load_param("./models/makeup.param");
    ret_bin = makeup_net.load_model("./models/makeup.bin");
    if (ret_param != 0 || ret_bin != 0) {
        fprintf(stderr,"makeup_net load param(%d), model(%d)\n", ret_param, ret_bin);
        return -1;
    }
    
    cv::Mat non_makeup_img = target_img.clone();
    cv::Mat makeup_img = reference_img.clone();

    //non-makeup parsing
    cv::Mat non_makeup_img_parsing;
    face_parsing(parsing_net, non_makeup_img, non_makeup_img_parsing);
    //makeup farsing
    cv::Mat makeup_img_parsing;
    face_parsing(parsing_net, makeup_img, makeup_img_parsing);
    //makeup transfer
    makeup_transfer(makeup_net, non_makeup_img, makeup_img, non_makeup_img_parsing, makeup_img_parsing);

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s [target_imagepath] [reference_imagepath]\n", argv[0]);
        return -1;
    }

    const char* target_imagepath = argv[1];
    //load target image
    cv::Mat target_img = cv::imread(target_imagepath, 1);
    if (target_img.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", target_imagepath);
        return -1;
    }

    const char* reference_imagepath = argv[2];
    //load reference image
    cv::Mat reference_img = cv::imread(reference_imagepath, 1);
    if (reference_img.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", reference_imagepath);
        return -1;
    }

    process(target_img, reference_img);
    
    return 0;
}
