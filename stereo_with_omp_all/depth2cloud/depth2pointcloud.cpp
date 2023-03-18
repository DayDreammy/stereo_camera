#include "depth2pointCloud.h"
#include <pcl/visualization/cloud_viewer.h>

// 定义点云类型
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// 相机内参
const double camera_factor = 1000;
const double camera_cx = 199;// 325.5;
const double camera_cy = 187;// 253.5;
const double camera_fx = 1500;// 518.0;
const double camera_fy = 1500;// 519.0;

void viewerOneOff(pcl::visualization::PCLVisualizer& viewer) {
	viewer.setBackgroundColor(1, 0.7, 1);
}


// 主函数
int main(int argc, char** argv)
{

	// 图像矩阵
	//cv::Mat rgb;
	// 使用cv::imread()来读取图像
	//rgb = cv::imread("RGB/0.335200.png");
	// rgb 图像是8UC3的彩色图像
	// depth 是16UC1的单通道图像，注意flags设置-1,表示读取原始数据不做任何修改
	Mat rgb = imread("im0.png", IMREAD_COLOR);
	Mat depth = imread("depth.png", IMREAD_COLOR);


	// 点云变量
	// 使用智能指针，创建一个空点云。这种指针用完会自动释放。
	PointCloud::Ptr cloud(new PointCloud);
	// 遍历深度图
	for (int m = 0; m < depth.rows; m++)
		for (int n = 0; n < depth.cols; n++)
		{
			// 获取深度图中(m,n)处的值
			unsigned short d = depth.ptr<unsigned short>(m)[n];
			// d 可能没有值，若如此，跳过此点
			if (d == 0)
				continue;
			// d 存在值，则向点云增加一个点
			PointT p;

			// 计算这个点的空间坐标
			p.z = -double(d) / camera_factor;
			p.x = -(n - camera_cx) * p.z / camera_fx;
			p.y = (m - camera_cy) * p.z / camera_fy;

			// 从rgb图像中获取它的颜色
			// rgb是三通道的BGR格式图，所以按下面的顺序获取颜色

			p.b = rgb.ptr<uchar>(m)[n * 3];
			p.g = rgb.ptr<uchar>(m)[n * 3 + 1];
			p.r = rgb.ptr<uchar>(m)[n * 3 + 2];

			// 把p加入到点云中
			cloud->points.push_back(p);
		}
	// 设置并保存点云
	cloud->height = 1;
	cloud->width = cloud->points.size();
	cout << "point cloud size = " << cloud->points.size() << endl;
	cloud->is_dense = false;

	pcl::io::savePCDFile("teddy.pcd", *cloud);
	pcl::visualization::CloudViewer view("Cloud Viewer");//直接创造一个显示窗口
	view.showCloud(cloud);//再这个窗口显示点云
	//view.runOnVisualizationThreadOnce(viewerOneOff);
	while (!view.wasStopped())
	{
	}

	// 清除数据并退出
	cloud->points.clear();
	cout << "Point cloud saved." << endl;
	return 0;
}
