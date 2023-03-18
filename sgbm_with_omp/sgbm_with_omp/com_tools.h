/*
 *  sgbm_types.h
 *  stereo_camera
 *
 *  Created by pp,yy on 1/6/22.
 *  Copyright 2022 Argus Corp. All rights reserved.
 *
 */
#pragma once
#include "sgbm_types.h"
#include "opencv2/opencv.hpp"
#include <iostream>
using namespace cv;
using namespace std;

/** @brief 均方根误差计算函数
* @param[in]  disp_e    估计得到的视差图
* @param[in]  disp_g    提供的标准视差图
* @return  eRMS
* @example double e_RMS = eRMS(disp8, dispg);
* @attention
* @note
*/
double eRMS(const Mat& disp_e, const Mat& disp_g);

/** @brief Percentage of error pixels(PEP)
 * @param[in]  disp_e    估计得到的视差图
 * @param[in]  disp_g    提供的标准视差图
 * @return  eRMS
 * @example double e_PEP = ePEP(disp8, dispg);
 * @attention
 * @note
 */
double ePEP(const Mat& disp_e, const Mat& disp_g);

/** @brief 均方根误差计算函数
* @param[in]  disp_e    估计得到的视差图
* @param[in]  disp_g    提供的标准视差图
* @return  eRMS
* @example double e_RMS = eRMS(disp8, dispg);
* @attention
* @note
*/
double eRMS(const Mat& disp_e, const Mat& disp_g);

/** @brief 视差图转深度图
* @param[in]   dispMap     视差图，8位单通道，CV_8UC1
* @param[out]  depthMap    深度图，16位无符号单通道，CV_16UC1
* @example                 disp2Depth(disp8, depth);
* @attention
* @note
*/

void disp2Depth(cv::Mat dispMap, cv::Mat& depthMap);


/** @brief census变换
 * @param[in]  source     图像
 * @param[out] census     census值数组
 * @param[in]  width      图像宽
 * @param[in]  height     图像高
 * @example
 */
void census_transform_rxr(const uint8* source, uint32* census, const sint32& width, const sint32& height, const sint32& size);
// Hamming距离
uint8 Hamming32(const uint32& x, const uint32& y);


/**
	 * \brief 左右路径聚合 → ←
	 * \param img_data			输入，影像数据
	 * \param width				输入，影像宽
	 * \param height			输入，影像高
	 * \param min_disparity		输入，最小视差
	 * \param max_disparity		输入，最大视差
	 * \param p1				输入，惩罚项P1
	 * \param p2_init			输入，惩罚项P2_Init
	 * \param cost_init			输入，初始代价数据
	 * \param cost_aggr			输出，路径聚合代价数据
	 * \param is_forward		输入，是否为正方向（正方向为从左到右，反方向为从右到左）
	 */
void CostAggregateLeftRight(const uint8* img_data, const sint32& width, const sint32& height, const sint32& min_disparity, const sint32& max_disparity,
	const sint32& p1, const sint32& p2_init, const uint8* cost_init, uint8* cost_aggr, bool is_forward = true);

/**
 * \brief 上下路径聚合 ↓ ↑
 * \param img_data			输入，影像数据
 * \param width				输入，影像宽
 * \param height			输入，影像高
 * \param min_disparity		输入，最小视差
 * \param max_disparity		输入，最大视差
 * \param p1				输入，惩罚项P1
 * \param p2_init			输入，惩罚项P2_Init
 * \param cost_init			输入，初始代价数据
 * \param cost_aggr			输出，路径聚合代价数据
 * \param is_forward		输入，是否为正方向（正方向为从上到下，反方向为从下到上）
 */
void CostAggregateUpDown(const uint8* img_data, const sint32& width, const sint32& height, const sint32& min_disparity, const sint32& max_disparity,
	const sint32& p1, const sint32& p2_init, const uint8* cost_init, uint8* cost_aggr, bool is_forward = true);



/**
	 * \brief 剔除小连通区
	 * \param disparity_map		输入，视差图
	 * \param width				输入，宽度
	 * \param height			输入，高度
	 * \param diff_insame		输入，同一连通区内的局部像素差异
	 * \param min_speckle_aera	输入，最小连通区面积
	 * \param invalid_val		输入，无效值
	 */
void RemoveSpeckles(float32* disparity_map, const sint32& width, const sint32& height, const sint32& diff_insame, const uint32& min_speckle_aera, const float32& invalid_val);


/**
 * \brief 中值滤波
 * \param in				输入，源数据
 * \param out				输出，目标数据
 * \param width				输入，宽度
 * \param height			输入，高度
 * \param wnd_size			输入，窗口宽度
 */
void MedianFilter(const float32* in, float32* out, const sint32& width, const sint32& height, const sint32 wnd_size);
