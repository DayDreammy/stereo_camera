#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/xfeatures2d/nonfree.hpp>
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

void KeyPointsToPoints(vector<KeyPoint>kpts, vector<Point2f>& pts)
{
    for (int i = 0; i < kpts.size(); i++)
    {
        pts.push_back(kpts[i].pt);
    }
    return;
}

int main()
{
    Mat srcImage1, srcImage2;
    srcImage1 = imread("00.jpg", 1);
    srcImage2 = imread("01.jpg", 1);
    //降采样原图,使大小一致
    float scale = 0.25;
    resize(srcImage1, srcImage1, cv::Size(), scale, scale, 1);
    resize(srcImage2, srcImage2, cv::Size(), scale, scale, 1);
    if (!srcImage1.data || !srcImage2.data)
    {
        cout << "读取出错" << endl;
        return false;
    }

    vector<KeyPoint>left_keypoints, right_keypoints;
    Ptr<FastFeatureDetector> ffd = FastFeatureDetector::create();
    ffd->detect(srcImage1, left_keypoints);
    ffd->detect(srcImage2, right_keypoints);

    vector<Point2f>left_points;
    KeyPointsToPoints(left_keypoints, left_points);

    vector<Point2f>right_points(left_points.size());
    KeyPointsToPoints(right_keypoints, right_points);

    Mat srcImage1_gray, srcImage2_gray;
    cvtColor(srcImage1, srcImage1_gray, CV_BGR2GRAY);
    cvtColor(srcImage2, srcImage2_gray, CV_BGR2GRAY);

    vector<uchar>vstatus;
    vector<float>verror;
    calcOpticalFlowPyrLK(srcImage1_gray, srcImage2_gray, left_points, right_points, vstatus, verror);

    Mat imofkl = srcImage1.clone();
    for (int i = 0; i < vstatus.size(); i++)
    {
        if (vstatus[i] && verror[i] < 12)
        {
            line(imofkl, left_points[i], right_points[i], CV_RGB(255, 255, 255), 1, 8, 0);
            circle(imofkl, right_points[i], 3, CV_RGB(255, 255, 255), 1, 8, 0);
        }
    }
    imshow("光流", imofkl);

    vector<Point2f>right_points_to_find;
    vector<int>right_points_to_find_back_index;
    for (unsigned int i = 0; i < vstatus.size(); i++)
    {
        if (vstatus[i] && verror[i] < 12)
        {
            right_points_to_find_back_index.push_back(i);
            right_points_to_find.push_back(right_points[i]);
        }
        else
        {
            vstatus[i] = 0;
        }
    }
    Mat right_points_to_find_flat = Mat(right_points_to_find).reshape(1, right_points_to_find.size());

    vector<Point2f>right_features;
    KeyPointsToPoints(right_keypoints, right_features);

    Mat right_features_flat = Mat(right_features).reshape(1, right_features.size());

    BFMatcher matcher(CV_L2);
    vector<vector<DMatch>>nearest_neighbors;
    matcher.radiusMatch(right_points_to_find_flat, right_features_flat, nearest_neighbors, 2.0f);

    set<int>found_in_right_points;
    vector<DMatch>matches;
    for (int i = 0; i < nearest_neighbors.size(); i++)
    {
        DMatch _m;
        if (nearest_neighbors[i].size() == 1)
        {
            _m = nearest_neighbors[i][0];
        }
        else if (nearest_neighbors[i].size() > 1)
        {
            double ratio = nearest_neighbors[i][0].distance / nearest_neighbors[i][1].distance;
            if (ratio < 0.7)
            {
                _m = nearest_neighbors[i][0];
            }
            else
            {
                continue;
            }
        }
        else
        {
            continue;
        }

        if (found_in_right_points.find(_m.trainIdx) == found_in_right_points.end())
        {
            _m.queryIdx = right_points_to_find_back_index[_m.queryIdx];
            matches.push_back(_m);
            found_in_right_points.insert(_m.trainIdx);
        }

    }
    cout << "pruned" << matches.size() << "/" << nearest_neighbors.size() << " matches" << endl;
    Mat result;
    drawMatches(srcImage1, left_keypoints, srcImage2, right_keypoints, matches, result);
    imshow("结果", result);

    /*   立体匹配    */
    
    // 提取特征点坐标
    vector<Point2f> img1P, img2P;
    for (int i = 0; i < matches.size(); i++)//将匹配的特征点坐标赋给point
    {
        int idx1 = matches.at(1).queryIdx;
        int idx2 = matches.at(1).trainIdx;

        img1P.push_back(left_keypoints.at(idx1).pt);
        img2P.push_back(right_keypoints.at(idx2).pt);

    }

    //
    //读入内参矩阵
    Mat K = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 摄像机内参数矩阵 */
    ifstream fin("cameraMatrix_m.txt"); /* 标定所用图像文件的路径 */
    //ifstream fin("cameraMatrix_m.txt"); /* 标定所用图像文件的路径 */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float a;
            fin >> a;
            K.at<float>(i, j) = a;
        }
    }
    // 读入畸变系数矩阵
    Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */
    ifstream fin2("distCoeffs.txt"); /* 标定所用图像文件的路径 */
    for (int i = 0; i < 5; i++) {
        float a;
        fin2 >> a;
        distCoeffs.at<float>(i) = a;
    }
    cout << " K : "<< K << endl;
    cout << " D : " << distCoeffs << endl;

    //通过特征点 得到基本矩阵E 
    Mat E = findEssentialMat(img1P, img2P, K, 8);
    cout << "E : " << E << endl;

    // 矩阵E分解，得到外参 旋转R和平移T 
    Mat R1, R2, t, R;
    decomposeEssentialMat(E, R1, R2, t);

    R = R1.clone();
    t = -t.clone();

    cout << " R :" << R << endl;
    cout << " t :" << t << endl;

    // 立体校正参数初始化
    Mat  P1, P2, Q;
    Rect validRoi[2];
    Mat mapLx, mapLy, mapRx, mapRy;     //映射表 
    Mat recImgL, recImgR;

    // 立体校正
    stereoRectify(K, distCoeffs, K, distCoeffs, srcImage1.size(), R, -R * t, R1, R2, P1, P2, Q);

    initUndistortRectifyMap(P1(cv::Rect(0, 0, 3, 3)), distCoeffs, R1, P1(cv::Rect(0, 0, 3, 3)), srcImage1.size(), CV_32FC1, mapLx, mapLy);
    remap(srcImage1, recImgL, mapLx, mapLy, 1);
    imwrite("recConyL.png", recImgL);

    initUndistortRectifyMap(P2(cv::Rect(0, 0, 3, 3)), distCoeffs, R2, P2(cv::Rect(0, 0, 3, 3)), srcImage2.size(), CV_32FC1, mapRx, mapRy);
    remap(srcImage2, recImgR, mapRx, mapRy, 1);
    imwrite("recConyR.png", recImgR);

    waitKey(0);
    return 0;
}