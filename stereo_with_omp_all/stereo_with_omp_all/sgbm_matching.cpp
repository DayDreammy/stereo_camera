/*
 *  sgbm_types.h
 *  stereo_camera
 *
 *  Created by pp,yy on 1/6/22.
 *  Copyright 2022 Argus Corp. All rights reserved.
 *
 */
 //#include "stdafx.h"
#include <iostream>
#include "sgbm_matching.h"
#include "com_tools.h"
#include <cassert>
#include <vector>
#include <algorithm>
#include <queue>
#include <opencv2/core/utility.hpp>

using namespace cv;
//#include "sgbm_types.h"


sgb_matching::sgb_matching()
{

}

sgb_matching::~sgb_matching()
{

	if (census_left_ != nullptr) {
		delete[] census_left_;
		census_left_ = nullptr;
	}
	if (census_right_ != nullptr) {
		delete[] census_right_;
		census_right_ = nullptr;
	}
	if (cost_init_ != nullptr) {
		delete[] cost_init_;
		cost_init_ = nullptr;
	}
	if (cost_aggr_ != nullptr) {
		delete[] cost_aggr_;
		cost_aggr_ = nullptr;
	}
	if (disp_left_ != nullptr) {
		delete[] disp_left_;
		disp_left_ = nullptr;
	}
	is_initialized_ = false;


}

bool sgb_matching::Initialize(const uint32& width, const uint32& height, const SGM_arg& arg)
{
	// ··· 赋值

	// 影像尺寸
	width_ = width;
	height_ = height;
	// SGM参数
	arg_ = arg;

	if (width == 0 || height == 0) {
		return false;
	}

	//··· 开辟内存空间

	// census值（左右图像）
	census_left_ = new uint32[width * height]();
	census_right_ = new uint32[width * height]();

	// 匹配代价（初始/聚合）
	const sint32 disp_range = arg.max_disparity - arg.min_disparity;
	if (disp_range <= 0) {
		return false;
	}
	cost_init_ = new uint8[width * height * disp_range]();
	cost_aggr_ = new uint16[width * height * disp_range]();

	cost_aggr_1_ = new uint8[width * height * disp_range]();
	cost_aggr_2_ = new uint8[width * height * disp_range]();
	cost_aggr_3_ = new uint8[width * height * disp_range]();
	cost_aggr_4_ = new uint8[width * height * disp_range]();
	cost_aggr_5_ = new uint8[width * height * disp_range]();
	cost_aggr_6_ = new uint8[width * height * disp_range]();
	cost_aggr_7_ = new uint8[width * height * disp_range]();
	cost_aggr_8_ = new uint8[width * height * disp_range]();

	// 视差图
	disp_left_ = new float32[width * height]();
	disp_right_ = new float32[width * height]();

	is_initialized_ = census_left_ && census_right_ && cost_init_ && cost_aggr_ && disp_left_;

	return is_initialized_;
}

bool sgb_matching::Match(const uint8* img_left, const uint8* img_right, float32* disp_left)
{
	if (!is_initialized_) {
		return false;
	}
	if (img_left == nullptr || img_right == nullptr) {
		return false;
	}

	img_left_ = img_left;
	img_right_ = img_right;
	block_size_ = arg_.block_size;

	// census变换
	census_transform();

	// 代价计算
	int64 t = getTickCount();
	compute_cost();
	t = getTickCount() - t;
	printf("Hamming cost %lf ms\n", t * 1000 / getTickFrequency());

	t = getTickCount();
	// 代价聚合
	cost_aggregation();

	t = getTickCount() - t;
	printf("cost aggregating! timing :	%lf ms\n", t * 1000 / getTickFrequency());


	// 视差计算
	compute_disparity();

	// 左右一致性检查
	if (arg_.is_check_lr) {
		// 视差计算（右影像）
		compute_disparity_right();
		// 一致性检查
		LR_check();
	}

	//移除小连通区
	if (arg_.is_remove_speckles) {
		RemoveSpeckles(disp_left_, width_, height_, 1, arg_.min_speckle_aera, invalid_float);
	}

	// 视差填充
	if (arg_.is_fill_holes) {
		FillHolesInDispMap();
	}

	//中值滤波
	MedianFilter(disp_left_, disp_left_, width_, height_, 3);

	// 输出视差图
	memcpy(disp_left, disp_left_, width_ * height_ * sizeof(float32));

	return true;
}

void sgb_matching::census_transform() const
{
	// 左右图像census变换
	census_transform_rxr(img_left_, census_left_, width_, height_, block_size_);
	census_transform_rxr(img_right_, census_right_, width_, height_, block_size_);
}

void sgb_matching::compute_cost() const
{

	const sint32& min_disparity = arg_.min_disparity;
	const sint32& max_disparity = arg_.max_disparity;
	const sint32 disp_range = max_disparity - min_disparity;

	// 计算代价（基于Hamming距离）
#pragma omp parallel for 
	for (sint32 i = 0; i < height_; i++) {
		for (sint32 j = 0; j < width_; j++) {

			// 左图census值
			const uint32 census_val_l = census_left_[i * width_ + j];

			// 逐视差计算代价值

			for (sint32 d = min_disparity; d < max_disparity; d++) {
				auto& cost = cost_init_[i * width_ * disp_range + j * disp_range + (d - min_disparity)];
				if (j - d < 0 || j - d >= width_) {
					cost = UINT8_MAX / 2;
					continue;
				}
				// 右图对应像点的census值
				const uint32 census_val_r = census_right_[i * width_ + j - d];

				// 计算匹配代价

				cost = Hamming32(census_val_l, census_val_r);

			}
		}
	}
}
void sgb_matching::compute_disparity() const
{
	// 最小最大视差
	const sint32& min_disparity = arg_.min_disparity;
	const sint32& max_disparity = arg_.max_disparity;
	const sint32 disp_range = max_disparity - min_disparity;

	if (disp_range <= 0) {
		return;
	}

	// 左影像视差图
	const auto disparity = disp_left_;
	// 左影像聚合代价数组
	const auto cost_ptr = cost_aggr_;
	// 未实现聚合步骤，暂用初始代价值来代替
   /* auto cost_ptr = cost_init_;*/

	const sint32 width = width_;
	const sint32 height = height_;

	// 为了加快读取效率，把单个像素的所有代价值存储到局部数组里
	std::vector<uint16> cost_local(disp_range);

	// 逐像素计算最优视差
	for (sint32 i = 0; i < height_; i++) {
		for (sint32 j = 0; j < width_; j++) {

			uint16 min_cost = UINT16_MAX;
			uint16 sec_min_cost = UINT16_MAX;
			uint16 max_cost = 0;
			sint32 best_disparity = 0;

			// 遍历视差范围内的所有代价值，输出最小代价值及对应的视差值
			for (sint32 d = min_disparity; d < max_disparity; d++) {
				const sint32 d_idx = d - min_disparity;
				const auto& cost = cost_ptr[i * width * disp_range + j * disp_range + d_idx];
				if (min_cost > cost) {
					min_cost = cost;
					best_disparity = d;
				}
				max_cost = std::max(max_cost, static_cast<uint16>(cost));
			}


			//最小代价值对应的视差值即为像素的最优视差
			if (max_cost != min_cost) {
				disp_left_[i * width_ + j] = static_cast<float>(best_disparity);
			}
			else {
				// 如果所有视差下的代价值都一样，则该像素无效
				disp_left_[i * width_ + j] = invalid_float;
			}


			// ---子像素拟合
			if (best_disparity == min_disparity || best_disparity == max_disparity - 1) {
				disparity[i * width + j] = invalid_float;
				continue;
			}
			// 最优视差前一个视差的代价值c_1，后一个视差的代价值c_2
			const sint32 idx_1 = best_disparity - 1 - min_disparity;
			const sint32 idx_2 = best_disparity + 1 - min_disparity;
			const uint16 c_1 = cost_local[idx_1];
			const uint16 c_2 = cost_local[idx_2];
			// 解一元二次曲线极值
			const uint16 den = std::max(1, c_1 + c_2 - 2 * min_cost);
			disparity[i * width + j] = static_cast<float32>(best_disparity) + static_cast<float32>(c_1 - c_2) / (den * 2.0f);
		}
	}
}


void sgb_matching::compute_disparity_right() const
{
	const sint32& min_disparity = arg_.min_disparity;
	const sint32& max_disparity = arg_.max_disparity;
	const sint32 disp_range = max_disparity - min_disparity;
	if (disp_range <= 0) {
		return;
	}

	// 右影像视差图
	const auto disparity = disp_right_;
	// 左影像聚合代价数组
	const auto cost_ptr = cost_aggr_;

	const sint32 width = width_;
	const sint32 height = height_;


	// 为了加快读取效率，把单个像素的所有代价值存储到局部数组里
	std::vector<uint16> cost_local(disp_range);

	// ---逐像素计算最优视差
	// 通过左影像的代价，获取右影像的代价
	// 右cost(xr,yr,d) = 左cost(xr+d,yl,d)
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			uint16 min_cost = UINT16_MAX;
			uint16 max_cost = 0;
			sint32 best_disparity = 0;

			// ---统计候选视差下的代价值
			for (sint32 d = min_disparity; d < max_disparity; d++) {
				const sint32 d_idx = d - min_disparity;
				const sint32 col_left = j + d;
				if (col_left >= 0 && col_left < width) {
					const auto& cost = cost_local[d_idx] = cost_ptr[i * width * disp_range + col_left * disp_range + d_idx];
					if (min_cost > cost) {
						min_cost = cost;
						best_disparity = d;
					}
				}
				else {
					cost_local[d_idx] = UINT16_MAX;
				}
			}

			// 最小代价值对应的视差值即为像素的最优视差
			if (max_cost != min_cost) {
				disp_right_[i * width_ + j] = static_cast<float>(best_disparity);
			}
			else {
				// 如果所有视差下的代价值都一样，则该像素无效
				disp_right_[i * width_ + j] = invalid_float;
			}

			// ---子像素拟合
			if (best_disparity == min_disparity || best_disparity == max_disparity - 1) {
				disparity[i * width + j] = invalid_float;
				continue;
			}
			// 最优视差前一个视差的代价值cost_1，后一个视差的代价值cost_2
			const sint32 idx_1 = best_disparity - 1 - min_disparity;
			const sint32 idx_2 = best_disparity + 1 - min_disparity;
			const uint16 cost_1 = cost_local[idx_1];
			const uint16 cost_2 = cost_local[idx_2];
			// 解一元二次曲线极值
			const uint16 denom = std::max(1, cost_1 + cost_2 - 2 * min_cost);
			disparity[i * width + j] = static_cast<float32>(best_disparity) + static_cast<float32>(cost_1 - cost_2) / (denom * 2.0f);
		}
	}
}



void sgb_matching::cost_aggregation() const
{
	// 路径聚合
	// 1、左->右/右->左
	// 2、上->下/下->上
	// 3、左上->右下/右下->左上
	// 4、右上->左上/左下->右上
	//
	// ↘ ↓ ↙   5  3  7
	// →    ←	 1    2
	// ↗ ↑ ↖   8  4  6
	//
	const auto& min_disparity = arg_.min_disparity;
	const auto& max_disparity = arg_.max_disparity;
	assert(max_disparity > min_disparity);

	const sint32 size = width_ * height_ * (max_disparity - min_disparity);
	if (size <= 0) {
		return;
	}

	const auto& P1 = arg_.p1;
	const auto& P2_Int = arg_.p2_int;

	if (arg_.num_paths == 4 || arg_.num_paths == 8) {
		// 左右聚合
		CostAggregateLeftRight(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_1_, true);
		CostAggregateLeftRight(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_2_, false);
		// 上下聚合
		CostAggregateUpDown(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_3_, true);
		CostAggregateUpDown(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_4_, false);
	}

	if (arg_.num_paths == 8) {
		// 对角线聚合
		//#TODO:完成对角线聚合代码

	}

	// 把4/8个方向加起来
	for (sint32 i = 0; i < size; i++) {
		if (arg_.num_paths == 4 || arg_.num_paths == 8) {
			cost_aggr_[i] = cost_aggr_1_[i] + cost_aggr_2_[i] + cost_aggr_3_[i] + cost_aggr_4_[i];
		}
		if (arg_.num_paths == 8) {
			cost_aggr_[i] += cost_aggr_5_[i] + cost_aggr_6_[i] + cost_aggr_7_[i] + cost_aggr_8_[i];
		}
	}
}


void sgb_matching::LR_check()
{
	const sint32 width = width_;
	const sint32 height = height_;

	const float32& threshold = arg_.lrcheck_thres;

	// 遮挡区像素和误匹配区像素
	auto& occlusions = occlusions_;
	auto& mismatches = mismatches_;
	occlusions.clear();
	mismatches.clear();

	// ---左右一致性检查

	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			// 左影像视差值
			auto& disp = disp_left_[i * width + j];
			if (disp == invalid_float) {
				mismatches.emplace_back(i, j);
				continue;
			}

			// 根据视差值找到右影像上对应的同名像素
			const auto col_right = static_cast<sint32>(j - disp + 0.5);//减去视差值

			if (col_right >= 0 && col_right < width) {
				// 右影像上同名像素的视差值
				const auto& disp_r = disp_right_[i * width + col_right];

				// 判断两个视差值是否一致（差值在阈值内）
				if (abs(disp - disp_r) > threshold) {
					// 区分遮挡区和误匹配区
					// 通过右影像视差算出在左影像的匹配像素，并获取视差disp_rl
					// if(disp_rl > disp) 
					//		pixel in occlusions
					// else 
					//		pixel in mismatches
					const sint32 col_rl = static_cast<sint32>(col_right + disp_r + 0.5);
					if (col_rl > 0 && col_rl < width) {
						const auto& disp_l = disp_left_[i * width + col_rl];
						if (disp_l > disp) {
							occlusions.emplace_back(i, j);
						}
						else {
							mismatches.emplace_back(i, j);
						}
					}
					else {
						mismatches.emplace_back(i, j);
					}

					// 让视差值无效
					disp = invalid_float;
				}
			}
			else {
				// 通过视差值在右影像上找不到同名像素（超出影像范围）
				disp = invalid_float;
				mismatches.emplace_back(i, j);
			}
		}
	}
}




void sgb_matching::FillHolesInDispMap()
{
	const sint32 width = width_;
	const sint32 height = height_;

	std::vector<float32> disp_collects;

	// 定义8个方向
	const float32 pi = 3.1415926f;
	float32 angle1[8] = { pi, 3 * pi / 4, pi / 2, pi / 4, 0, 7 * pi / 4, 3 * pi / 2, 5 * pi / 4 };
	float32 angle2[8] = { pi, 5 * pi / 4, 3 * pi / 2, 7 * pi / 4, 0, pi / 4, pi / 2, 3 * pi / 4 };
	float32* angle = angle1;
	// 最大搜索行程，没有必要搜索过远的像素
	const sint32 max_search_length = 1.0 * std::max(abs(arg_.max_disparity), abs(arg_.min_disparity));

	float32* disp_ptr = disp_left_;

	for (sint32 k = 0; k < 3; k++) {
		// 第一次循环处理遮挡区，第二次循环处理误匹配区
		auto& trg_pixels = (k == 0) ? occlusions_ : mismatches_;
		if (trg_pixels.empty()) {
			continue;
		}
		std::vector<float32> fill_disps(trg_pixels.size());
		std::vector<std::pair<sint32, sint32>> inv_pixels;
		if (k == 2) {
			//  第三次循环处理前两次没有处理干净的像素
			for (sint32 i = 0; i < height; i++) {
				for (sint32 j = 0; j < width; j++) {
					if (disp_ptr[i * width + j] == invalid_float) {
						inv_pixels.emplace_back(i, j);
					}
				}
			}
			trg_pixels = inv_pixels;
		}

		// 遍历待处理像素

		for (auto n = 0u; n < trg_pixels.size(); n++) {
			auto& pix = trg_pixels[n];
			const sint32 y = pix.first;
			const sint32 x = pix.second;

			if (y == height / 2) {
				angle = angle2;
			}

			// 收集8个方向上遇到的首个有效视差值
			disp_collects.clear();
			for (sint32 s = 0; s < 8; s++) {
				const float32 ang = angle[s];
				const float32 sina = float32(sin(ang));
				const float32 cosa = float32(cos(ang));
				for (sint32 m = 1; m < max_search_length; m++) {
					const sint32 yy = lround(y + m * sina);
					const sint32 xx = lround(x + m * cosa);
					if (yy < 0 || yy >= height || xx < 0 || xx >= width) {
						break;
					}
					const auto& disp = *(disp_ptr + yy * width + xx);
					if (disp != invalid_float) {
						disp_collects.push_back(disp);
						break;
					}
				}
			}
			if (disp_collects.empty()) {
				continue;
			}

			std::sort(disp_collects.begin(), disp_collects.end());

			// 如果是遮挡区，则选择第二小的视差值
			// 如果是误匹配区，则选择中值
			if (k == 0) {
				if (disp_collects.size() > 1) {
					fill_disps[n] = disp_collects[1];
				}
				else {
					fill_disps[n] = disp_collects[0];
				}
			}
			else {
				fill_disps[n] = disp_collects[disp_collects.size() / 2];
			}
		}
		for (auto n = 0u; n < trg_pixels.size(); n++) {
			auto& pix = trg_pixels[n];
			const sint32 y = pix.first;
			const sint32 x = pix.second;
			disp_ptr[y * width + x] = fill_disps[n];
		}
	}
}
