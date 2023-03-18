#include <iostream>
#include <string>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
using namespace std;
using namespace cv;

void depth2pointcloud(const Mat& depth,const Mat& rgb);
