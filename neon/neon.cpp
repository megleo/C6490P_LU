#include "convert.hpp"
#include <arm_neon.h>
#include <iostream>
void convert_reference(const uint8_t * src, uint8_t * dst, const int n);
void convert_neon(const uint8_t * src, uint8_t * dst, const int n);

void convert_reference(const uint8_t * src, uint8_t * dst, const int n)
{
	for(int i = 0; i < n; i++)
	{
		int r = *src++; // load red
		int g = *src++; // load green
		int b = *src++; // load blue

		// build weighted average:
		int y = (r*77)+(g*151)+(b*28);

		// undo the scale by 256 and write to memory:
		*dst++ = (y>>8);
	}
}

void convert_neon(const uint8_t * src, uint8_t * dst, const int n)
{
	const uint8x8_t rfac = vdup_n_u8(77);
	const uint8x8_t gfac = vdup_n_u8(151);
	const uint8x8_t bfac = vdup_n_u8(28);
	const int n_ = n >> 3;

	for(int i = 0; i < n_; i++)
	{
		uint16x8_t temp;
		uint8x8x3_t rgb = vld3_u8(src);
		uint8x8_t result;

		temp = vmull_u8(rgb.val[0], rfac);
		temp = vmlal_u8(temp,rgb.val[1], gfac);
		temp = vmlal_u8(temp,rgb.val[2], bfac);
		result = vshrn_n_u16(temp, 8);
		vst1_u8(dst, result);

		src  += 8*3;
		dst += 8;
	}
}

#include "convert.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

bool verify(const cv::Mat& img1, const cv::Mat& img2)
{
	cv::Mat diff;
	cv::absdiff(img1, img2, diff);
	return (cv::countNonZero(diff) == 0);
}

double launch_convert_reference(const cv::Mat& src, cv::Mat& dst, const int pixel_num, const int loop_num)
{
	cv::TickMeter tm;
	double time = 0.0;
	for(int i = 0; i <= loop_num; i++)
	{
		tm.reset();
		tm.start();
		convert_reference((uint8_t *)src.datastart, (uint8_t*)dst.datastart, pixel_num);
		tm.stop();
		time += ((i>0) ? 0.0 : tm.getTimeMilli());
	}
	return time;
}

double launch_convert_neon(const cv::Mat& src, cv::Mat& dst, const int pixel_num, const int loop_num)
{
	cv::TickMeter tm;
	double time = 0.0;
	for(int i = 0; i < loop_num; i++)
	{
		tm.reset();
		tm.start();
		convert_neon((uint8_t *)src.datastart, (uint8_t*)dst.datastart, pixel_num);
		tm.stop();
		time += ((i>0) ? 0.0 : tm.getTimeMilli());
	}
	return time;
}


int main(int argc, char *argv[])
{
	cv::Mat src = cv::imread("lena.jpg", cv::IMREAD_COLOR);
	if (src.empty())
	{
		std::cout << "could not load image." << std::endl;
		return -1;
	}

	cv::resize(src, src, cv::Size(4096, 4096));
	cv::cvtColor(src, src, cv::COLOR_BGR2RGB);
	const int pixel_num = src.cols * src.rows;

	cv::Mat dst(src.size(), CV_8UC1, cv::Scalar(0));
	cv::Mat dst_neon(src.size(), CV_8UC1, cv::Scalar(0));

	const int loop_num = 10;
	double time_ref = launch_convert_reference(src, dst, pixel_num, loop_num);
	double time_neon = launch_convert_neon(src, dst_neon, pixel_num, loop_num);

	std::cout << "time(reference) = " << time_ref << std::endl;
	std::cout << "time(NEON) = " << time_neon << std::endl;
	std::cout << "verify: " << (verify(dst, dst_neon) ? "Passed" : "Failed") << std::endl;

	return 0;
}