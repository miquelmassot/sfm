#ifndef SFM_H
#define SFM_H

// system includes
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>

// ROS includes
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_geometry/pinhole_camera_model.h>
#include <camera_calibration_parsers/parse_yml.h>
#include <tf/tf.h>

// Package includes

namespace sfm{
class StructureFromMotion
{
public:
	StructureFromMotion(std::vector<cv::Mat> imgs);
	void setCameraInfo(const sensor_msgs::CameraInfoConstPtr& cam_info);

private:
	void init();

	pcl::PointCloud<pcl::PointXYZRGB> pointcloud_;

	// Params
	int min_matches_;
	int min_kp_;
	int bucket_width_;
	int bucket_height_;
	int max_bucket_features_;
	double max_sq_displacement_;

	// ROS related
	ros::NodeHandle nh_;
	ros::NodeHandle nhp_;

	// Images
	std::vector<cv::Mat> imgs_;
	int curr_idx_;

	bool first_image_;
	cv::Mat prev_img_;
	cv::Mat curr_img_;

	tf::StampedTransform prev_pose_;
  std::vector<cv::DMatch> filt_matches_;
  cv::Ptr<cv::FeatureDetector> detector_;
  cv::Ptr<cv::DescriptorExtractor> descriptor_extractor_;
  cv::Ptr<cv::DescriptorMatcher> descriptor_matcher_;
  std::vector<cv::KeyPoint> prev_kp_;
  std::vector<cv::KeyPoint> curr_kp_;
  std::vector<cv::Point3f> prev_pt_3d_;
  std::vector<cv::Point2f> curr_pt_2d_;
	cv::Mat prev_desc_;
	cv::Mat curr_desc_;
	double matching_threshold_;

	sensor_msgs::CameraInfo cam_info_;
	image_geometry::PinholeCameraModel pcm_;

	bool process();

	void Triangulate(const cv::Point2f& pt1, 
									 const cv::Point2f& pt2, 
									 const cv::Mat& K, 
									 const cv::Mat& Kinv, 
									 const cv::Mat& P1, 
									 const cv::Mat& P2, 
									 cv::Point3d& xyzPoint, 
									 bool debug);

	void TriangulatePoints(const std::vector<cv::Point2f>& pt_set1,
	                       const std::vector<cv::Point2f>& pt_set2,
	                       const cv::Mat& K,
	                       const cv::Mat& Kinv,
	                       const cv::Mat& P1,
	                       const cv::Mat& P2,
	                       std::vector<cv::Point3d>& pointcloud);

	void IterativeLinearLSTriangulation(      cv::Mat&    dst, 
																			const cv::Point3d& u1, 
																			const cv::Mat&     P1, 
																			const cv::Point3d& u2, 
																			const cv::Mat&     P2);

	void LinearLSTriangulation(      cv::Mat&    dst, 
											 			 const cv::Point3d& u1, 
										 				 const cv::Mat&     P1, 
								  	 				 const cv::Point3d& u2, 
								 						 const cv::Mat&     P2);

	static void keypointDetector( const cv::Mat& image, 
				                        std::vector<cv::KeyPoint>& key_points, 
				                        std::string type);
	static void descriptorExtraction(const cv::Mat& image,
 																	 std::vector<cv::KeyPoint>& key_points, 
 																	 cv::Mat& descriptors, std::string type);
	static void thresholdMatching(const cv::Mat& descriptors1, 
																const cv::Mat& descriptors2,
	  														double threshold, 
	  														const cv::Mat& match_mask, 
	  														std::vector<cv::DMatch>& matches);
	static void crossCheckFilter(const std::vector<cv::DMatch>& matches1to2, 
													     const std::vector<cv::DMatch>& matches2to1,
													     std::vector<cv::DMatch>& checked_matches);
	static void crossCheckThresholdMatching(const cv::Mat& descriptors1, 
																					const cv::Mat& descriptors2, 
																					double threshold, 
																					const cv::Mat& match_mask, 
																					std::vector<cv::DMatch>& matches);
	static std::vector<cv::KeyPoint> bucketKeypoints(const std::vector<cv::KeyPoint>& kp, 
					                                         int b_width, 
					                                         int b_height, 
					                                         int b_num_feautres);
	static void projectionToTransformation(const cv::Mat& P, cv::Mat& C);
	static void transformationToProjection(const cv::Mat& C, cv::Mat& P);
};
};

#endif