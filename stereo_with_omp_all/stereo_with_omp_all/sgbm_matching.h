/*
 *  sgbm_types.h
 *  stereo_camera
 *
 *  Created by pp,yy on 1/6/22.
 *  Copyright 2022 Argus Corp. All rights reserved.
 *
 */

#include "sgbm_types.h"
#include <omp.h>
#include <vector>
#pragma once

class sgb_matching
{
public:
	sgb_matching();
	~sgb_matching();


	/** @brief SGM参数结构体 */
	struct SGM_arg {
		uint8	num_paths;		// 聚合路径数
		sint32  min_disparity;	// 最小视差
		sint32	max_disparity;	// 最大视差
		sint32	block_size;  	// 方框大小

		// P1,P2 
		// P2 = P2_int / (Ip-Iq)
		sint32  p1;				// 惩罚项参数P1
		sint32  p2_int;			// 惩罚项参数P2

		//优化
		bool	is_check_unique;	// 是否检查唯一性
		float32	uniqueness_ratio;	// 唯一性约束阈值 （最小代价-次最小代价)/最小代价 > 阈值 为有效像素

		bool	is_check_lr;		// 是否检查左右一致性
		float32	lrcheck_thres;		// 左右一致性约束阈值

		bool	is_remove_speckles;	// 是否移除小的连通区
		int		min_speckle_aera;	// 最小的连通区面积（像素数）

		bool	is_fill_holes;		// 是否填充视差空洞

		SGM_arg() : num_paths(4), min_disparity(0), max_disparity(64), 
			block_size(5),is_check_unique(true), uniqueness_ratio(0.95f),
			is_check_lr(true), lrcheck_thres(1.0f),
			is_remove_speckles(true), min_speckle_aera(30),
			is_fill_holes(true), p1(10), p2_int(150) {
		}

	};

public:
	/** @brief 类的初始化，完成一些内存的预分配、参数的预设置等
	* @param[in]  width     图像宽度
	* @param[in]  height    图像高度
	* @param[in]  arg       sgb_matching参数
	* @example S_D(img1, img2, disp, 3,1);
	*/
	bool Initialize(const uint32& width, const uint32& height, const SGM_arg& arg);

	/** @brief 匹配函数
	* @param[in]  img_left     左图像指针
	* @param[in]  img_right    右图像指针
	* @param[out]  disp_left   左影像视差图指针
	* @example S_D(img1, img2, disp, 3,1);
	*/
	bool Match(const uint8* img_left, const uint8* img_right, float32* disp_left);

private:
	/** @brief Census变换 */
	void census_transform() const;

	/** @brief 代价计算	 */
	void compute_cost() const;

	/** @brief 代价聚合	 */
	void cost_aggregation() const;

	/** @brief 视差计算	 */
	void compute_disparity() const;
	void compute_disparity_right() const;
	
	/** @brief 一致性检查 */
	void LR_check() ;

	/** \brief 视差图填充 */
	void FillHolesInDispMap();


private:
	/** @brief SGM参数	 */
	SGM_arg arg_;

	/** @brief 图像宽	 */
	sint32 width_;

	/** @brief 图像高	 */
	sint32 height_;

	/** @brief 方框大小	 */
	sint32 block_size_;

	/** @brief 左图像数据	 */
	const uint8* img_left_;

	/** @brief 右图像数据	 */
	const uint8* img_right_;

	/** @brief 左图像census值	*/
	uint32* census_left_;

	/** @brief 右图像census值	*/
	uint32* census_right_;

	/** @brief 初始匹配代价	*/
	uint8* cost_init_;

	/** @brief 聚合匹配代价	*/

	uint16* cost_aggr_;

	// ↘ ↓ ↙   5  3  7
	// →    ←	 1    2
	// ↗ ↑ ↖   8  4  6
	/** \brief 聚合匹配代价-方向1	*/
	uint8* cost_aggr_1_;
	/** \brief 聚合匹配代价-方向2	*/
	uint8* cost_aggr_2_;
	/** \brief 聚合匹配代价-方向3	*/
	uint8* cost_aggr_3_;
	/** \brief 聚合匹配代价-方向4	*/
	uint8* cost_aggr_4_;
	/** \brief 聚合匹配代价-方向5	*/
	uint8* cost_aggr_5_;
	/** \brief 聚合匹配代价-方向6	*/
	uint8* cost_aggr_6_;
	/** \brief 聚合匹配代价-方向7	*/
	uint8* cost_aggr_7_;
	/** \brief 聚合匹配代价-方向8	*/
	uint8* cost_aggr_8_;


	/** @brief 左图像视差图	*/
	float32* disp_left_;

	/** @brief 右图像视差图	*/
	float32* disp_right_;

	/** @brief 是否初始化标志	*/
	bool is_initialized_;



	/** @brief 遮挡区像素集	*/
	std::vector<std::pair<int, int>> occlusions_;
	/** @brief 误匹配区像素集	*/
	std::vector<std::pair<int, int>> mismatches_;

};
