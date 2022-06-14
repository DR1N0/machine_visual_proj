#include <QtCore/QCoreApplication>
#include <iostream>
#include <string>
using namespace std;
 
// OpenCV 库
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
 
// PCL 库
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
 
// 定义点云类型
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;
 
// 相机内参
const double camera_factor = 0.00012498664727900177;
const double camera_cx = 312.83380126953125;
const double camera_cy = 241.61764526367188;
const double camera_fx = 622.0875244140625;
const double camera_fy = 622.0875854492188;

int main(int argc, char *argv[])
{
	std::string name = "454";
	std::string img_name = "out/"+name+".jpg";
	std::string depth_name = "out/"+name+".png";
	std::string pcd_name = "out/"+name+".pcd";
	// QCoreApplication a(argc, argv);
	cv::Mat rgb, depth;
	// 使用cv::imread()来读取图像
	
	rgb = cv::imread(img_name);
	// rgb 图像是8UC3的彩色图像
	// depth 是16UC1的单通道图像，注意flags设置-1,表示读取原始数据不做任何修改
	depth = cv::imread(depth_name, -1);
 
	// 点云变量
	// 使用智能指针，创建一个空点云。这种指针用完会自动释放。
	PointCloud::Ptr cloud(new PointCloud);
	// 遍历深度图
	for (int m = 0; m < depth.rows; m++)
		for (int n = 0; n < depth.cols; n++)
		{
			// 获取深度图中(m,n)处的值
			ushort d = depth.ptr<ushort>(m)[n];
			// d 可能没有值，若如此，跳过此点
			if (d == 0)
				continue;
			// d 存在值，则向点云增加一个点
			PointT p;
 
			// 计算这个点的空间坐标
			p.z = double(d) * camera_factor;
			p.x = (n - camera_cx) * p.z / camera_fx;
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
	pcl::io::savePCDFile(pcd_name, *cloud);
	// 清除数据并退出
	cloud->points.clear();
	cout << "Point cloud saved." << endl;
	// return a.exec();
	return 1;
}