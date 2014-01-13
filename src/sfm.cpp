#include <sfm/sfm.h>
#include <opencv2/nonfree/nonfree.hpp>

#define EPSILON 1e-6

sfm::StructureFromMotion::StructureFromMotion(std::vector<cv::Mat> imgs): imgs_(imgs)
{
	init();
}

void sfm::StructureFromMotion::init()
{
	nh_ = ros::NodeHandle();
	nhp_ = ros::NodeHandle("~");
	curr_idx_ = 0;
	first_image_ = true;

	ROS_INFO_STREAM("[SFM]: " << imgs_.size() << " images entered.");

	std::string camera_info_yaml,camera_name;
  nhp_.param("camera_info_yaml", camera_info_yaml, std::string("calibration.yaml"));
  nhp_.param("camera_name", camera_name, std::string("left_camera"));

	// Load the camera info yaml file
  camera_calibration_parsers::readCalibrationYml(camera_info_yaml, camera_name, cam_info_);
  pcm_.fromCameraInfo(cam_info_);
  ROS_INFO_STREAM("[SFM]: Camera info YAML file loaded correctly.");

	// Init params
	nhp_.param("min_keypoints",min_kp_, 20);
	nhp_.param("min_matches",min_matches_, 8);
	nhp_.param("matching_threshold", matching_threshold_, 0.85);
	nhp_.param("bucket_width", bucket_width_, 300);
	nhp_.param("bucket_height", bucket_height_, 300);
	nhp_.param("max_bucket_features", max_bucket_features_, 500);
	nhp_.param("max_sq_displacement", max_sq_displacement_, 2000.0);

	ROS_INFO_STREAM("[SFM]: Processing...");
	while(curr_idx_<imgs_.size())
		process();
}

bool sfm::StructureFromMotion::process()
{
	// Get a new image
	cv::Mat raw = imgs_[curr_idx_].clone();
  cv::Mat bw(raw.size(),CV_8UC1);
  cv::cvtColor( raw, bw, CV_BGR2GRAY );
	pcm_.rectifyImage(bw,curr_img_);

  cv::equalizeHist(curr_img_,curr_img_);

	// Test if it's the first time
	if(first_image_)
	{
		ROS_INFO("[SFM]: First image received.");
		prev_img_ = curr_img_;
		keypointDetector(prev_img_,prev_kp_,"SIFT");
		prev_desc_ = cv::Mat_<std::vector<float> >();
		descriptorExtraction(prev_img_,prev_kp_,prev_desc_,"SIFT");
		prev_kp_ = bucketKeypoints(prev_kp_, bucket_width_, bucket_height_, max_bucket_features_);
		ROS_INFO_STREAM("[SFM]: Found " << prev_kp_.size() << " keypoints in image " << curr_idx_); 
		first_image_ = false;
		curr_idx_++;
		return false;
	}

	// Extract keypoints and descriptors of images
  curr_desc_ = cv::Mat_<std::vector<float> >();
  keypointDetector(curr_img_, curr_kp_, "SIFT");

  curr_kp_ = bucketKeypoints(curr_kp_, bucket_width_, bucket_height_, max_bucket_features_);

  if(curr_kp_.size() < min_kp_ )
  {
  	ROS_INFO_STREAM("[SFM]: Not enough keypoints: " << curr_kp_.size() << " and min is " << min_kp_);
  	curr_idx_++;
  	return false;
  }
  ROS_INFO_STREAM("[SFM]: Found " << curr_kp_.size() << " keypoints in image " << curr_idx_); 

  // Extract the descriptors
  descriptorExtraction(curr_img_, curr_kp_, curr_desc_, "SIFT");

  // Find matching between stereo images
  std::vector<cv::DMatch> matches;
  cv::Mat mask;
  crossCheckThresholdMatching(prev_desc_, curr_desc_, matching_threshold_, mask,  matches);
  ROS_INFO_STREAM("[SFM]: Found " << matches.size() << " matches between images " << curr_idx_ - 1 << " and " << curr_idx_); 

  std::vector<cv::DMatch> matches_filtered;
  // Filter matches by allowed displacement 
  for (size_t i = 0; i < matches.size(); ++i)
  {
  	double my = prev_kp_[matches[i].queryIdx].pt.y - curr_kp_[matches[i].trainIdx].pt.y;
  	double mx = prev_kp_[matches[i].queryIdx].pt.x - curr_kp_[matches[i].trainIdx].pt.x;
    //if ( mx*mx+my*my < max_sq_displacement_)
      matches_filtered.push_back(matches[i]);
  }


  if(matches_filtered.size() < min_matches_)
  {
  	ROS_INFO_STREAM("[SFM]: Not enough matches: " << matches_filtered.size() << " and min is " << min_matches_);
  	curr_idx_++;
  	return false;
  }
  ROS_INFO_STREAM("[SFM]: Found " << matches_filtered.size() << " filtered matches between images " << curr_idx_ - 1 << " and " << curr_idx_); 

  // show the matches
  cv::Mat matches_img;
  drawMatches(prev_img_, prev_kp_, curr_img_, curr_kp_, matches_filtered, matches_img);
  std::string winname = "Matches";
  cv::namedWindow(winname,CV_WINDOW_NORMAL);
  cv::imshow(winname,matches_img);
  cv::waitKey(5);

  // Extract matched points 
  std::vector<cv::Point2f> prev_pt,curr_pt;

  for (size_t i = 0; i < matches_filtered.size(); ++i)
  {
    int prev_idx = matches_filtered[i].queryIdx;
    int curr_idx = matches_filtered[i].trainIdx;
    prev_pt.push_back(prev_kp_[prev_idx].pt);
    curr_pt.push_back(curr_kp_[curr_idx].pt);
  }

  cv::Mat status;
  // get the camera matrix
  cv::Mat K = pcm_.intrinsicMatrix();
	// Find (R|t) of the current camera pose respect to the prev pose.
	cv::Mat F = cv::findFundamentalMat(prev_pt, curr_pt, CV_FM_RANSAC, 0.1, 0.99, status);
	cv::Mat E = K.t() * F * K; //according to HZ (9.12)

	// Solve the system
	cv::SVD svd(E);
	cv::Mat W = (cv::Mat_<double>(3,3) << 0,-1,0,1,0,0,0,0,1);
	//cv::Mat Winv = (cv::Mat_<double>(3,3) << 0,1,0,-1,0,0,0,0,1);
	cv::Mat R = svd.u * W * svd.vt; // Rotation
	cv::Mat t = svd.u.col(2); //u3 translation
	cv::Mat P2 = (cv::Mat_<double>(3,4) << 
		R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0), 
		R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1),
    R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2));
	cv::Mat P1 = (cv::Mat_<double>(3,4) << 
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0);

	ROS_INFO_STREAM("Found R|t as \nR:\n" << R << "\nt:\n" << t);

	pcl::PointCloud<pcl::PointXYZRGB> coud_in,cloud_out;
	cv::Mat Kinv = K.inv();
	TriangulatePoints( prev_pt, curr_pt, K, Kinv, P1, P2, pointcloud);

  Eigen::Affine3f transform;
  pcl::transformPointCloud(coud_in, cloud_out, transform);

  pointcloud_ += cloud_out;
	
	curr_idx_++;
}

/** \brief extract the keypoints of some image
  * @return 
  * \param image the source image
  * \param key_points is the pointer for the resulting image key_points
  * \param type descriptor type (see opencv docs)
  */
void sfm::StructureFromMotion::keypointDetector(const cv::Mat& image, 
																								std::vector<cv::KeyPoint>& key_points, 
																								std::string type)
{
  cv::initModule_nonfree();
  cv::Ptr<cv::FeatureDetector> cv_detector;
  cv_detector = cv::FeatureDetector::create(type);
  try
  {
    cv_detector->detect(image, key_points);
  }
  catch (cv::Exception& e)
  {
    ROS_WARN("[SFM]: cv_detector exception: %s", e.what());
  }
}

/** \brief extract descriptors of some image
  * @return 
  * \param image the source image
  * \param key_points keypoints of the source image
  * \param descriptors is the pointer for the resulting image descriptors
  */
void sfm::StructureFromMotion::descriptorExtraction(const cv::Mat& image,
													 													std::vector<cv::KeyPoint>& key_points, 
													 													cv::Mat& descriptors, std::string type)
{
  cv::Ptr<cv::DescriptorExtractor> cv_extractor;
  cv_extractor = cv::DescriptorExtractor::create(type);
  try
  {
    cv_extractor->compute(image, key_points, descriptors);
  }
  catch (cv::Exception& e)
  {
    ROS_WARN("[SFM]: cv_extractor exception: %s", e.what());
  }
}

/** \brief match descriptors of 2 images by threshold
  * @return 
  * \param descriptors1 descriptors of image1
  * \param descriptors2 descriptors of image2
  * \param threshold to determine correct matchings
  * \param match_mask mask for matchings
  * \param matches output vector with the matches
  */
void sfm::StructureFromMotion::thresholdMatching(const cv::Mat& descriptors1, 
																						 		 const cv::Mat& descriptors2,
																						 		 double threshold, 
																								 const cv::Mat& match_mask, 
																								 std::vector<cv::DMatch>& matches)
{
  matches.clear();
  if (descriptors1.empty() || descriptors2.empty())
    return;
  assert(descriptors1.type() == descriptors2.type());
  assert(descriptors1.cols == descriptors2.cols);

  const int knn = 2;
  cv::Ptr<cv::DescriptorMatcher> descriptor_matcher;
  // choose matcher based on feature type
  if (descriptors1.type() == CV_8U)
  {
    descriptor_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
  }
  else
  {
    descriptor_matcher = cv::DescriptorMatcher::create("BruteForce");
  }
  std::vector<std::vector<cv::DMatch> > knn_matches;
  descriptor_matcher->knnMatch(descriptors1, descriptors2,
          knn_matches, knn);

  for (size_t m = 0; m < knn_matches.size(); m++ )
  {
    if (knn_matches[m].size() < 2) continue;
    bool match_allowed = match_mask.empty() ? true : match_mask.at<unsigned char>(
        knn_matches[m][0].queryIdx, knn_matches[m][0].trainIdx) > 0;
    float dist1 = knn_matches[m][0].distance;
    float dist2 = knn_matches[m][1].distance;
    if (dist1 / dist2 < threshold && match_allowed)
    {
      matches.push_back(knn_matches[m][0]);
    }
  }
}

/** \brief filter matches of cross check matching
  * @return 
  * \param matches1to2 matches from image 1 to 2
  * \param matches2to1 matches from image 2 to 1
  * \param matches output vector with filtered matches
  */
void sfm::StructureFromMotion::crossCheckFilter(const std::vector<cv::DMatch>& matches1to2, 
																						    const std::vector<cv::DMatch>& matches2to1,
																						    std::vector<cv::DMatch>& checked_matches)
{
  checked_matches.clear();
  for (size_t i = 0; i < matches1to2.size(); ++i)
  {
    bool match_found = false;
    const cv::DMatch& forward_match = matches1to2[i];
    for (size_t j = 0; j < matches2to1.size() && match_found == false; ++j)
    {
      const cv::DMatch& backward_match = matches2to1[j];
      if (forward_match.trainIdx == backward_match.queryIdx &&
          forward_match.queryIdx == backward_match.trainIdx)
      {
        checked_matches.push_back(forward_match);
        match_found = true;
      }
    }
  }
}

/** \brief match descriptors of 2 images by threshold
  * @return 
  * \param descriptors1 descriptors of image 1
  * \param descriptors2 descriptors of image 2
  * \param threshold to determine correct matchings
  * \param match_mask mask for matchings
  * \param matches output vector with the matches
  */
void sfm::StructureFromMotion::crossCheckThresholdMatching(const cv::Mat& descriptors1, 
																													 const cv::Mat& descriptors2, 
																													 double threshold, 
																													 const cv::Mat& match_mask, 
																													 std::vector<cv::DMatch>& matches)
{
  std::vector<cv::DMatch> query_to_train_matches;
  thresholdMatching(descriptors1, descriptors2, threshold, match_mask, query_to_train_matches);
  std::vector<cv::DMatch> train_to_query_matches;
  cv::Mat match_mask_t;
  if (!match_mask.empty()) match_mask_t = match_mask.t();
  thresholdMatching(descriptors2, descriptors1, threshold, match_mask_t, train_to_query_matches);

  crossCheckFilter(query_to_train_matches, train_to_query_matches, matches);
}

/** \brief Keypoints bucketing
  * @return vector of keypoints after bucketing filtering
  * \param kp vector of keypoints
  * \param b_width is the width of the buckets
  * \param b_height is the height of the buckets
  * \param b_num_feautres is the maximum number of features per bucket
  */
std::vector<cv::KeyPoint> sfm::StructureFromMotion::bucketKeypoints(const std::vector<cv::KeyPoint>& kp, 
														                                        int b_width, 
														                                        int b_height, 
														                                        int b_num_feautres)
{
  // Find max values
  float x_max = 0;
  float y_max = 0;
  for (size_t i=0; i<kp.size(); i++)
  {
    if (kp[i].pt.x > x_max) x_max = kp[i].pt.x;
    if (kp[i].pt.y > y_max) y_max = kp[i].pt.y;
  }

  // Allocate number of buckets needed
  int bucket_cols = (int)floor(x_max/b_width) + 1;
  int bucket_rows = (int)floor(y_max/b_height) + 1;
  std::vector<cv::KeyPoint> *buckets = new std::vector<cv::KeyPoint>[bucket_cols*bucket_rows];

  // Assign keypoints to their buckets
  for (size_t i=0; i<kp.size(); i++)
  {
    int u = (int)floor(kp[i].pt.x/b_width);
    int v = (int)floor(kp[i].pt.y/b_height);
    buckets[v*bucket_cols+u].push_back(kp[i]);
  }

  // Refill keypoints from buckets
  std::vector<cv::KeyPoint> output;
  for (int i=0; i<bucket_cols*bucket_rows; i++)
  {
    // Sort keypoints by response
    //sort(buckets[i].begin(), buckets[i].end(), stereo_slam::OpencvUtils::sortKpByResponse);

    // shuffle bucket indices randomly
    random_shuffle(buckets[i].begin(),buckets[i].end());
    
    // Add up to max_features features from this bucket to output
    int k=0;
    for (std::vector<cv::KeyPoint>::iterator it=buckets[i].begin(); it!=buckets[i].end(); it++)
    {
      output.push_back(*it);
      k++;
      if (k >= b_num_feautres)
        break;
    }
  }
  return output;
}



void sfm::StructureFromMotion::TriangulatePoints(const std::vector<cv::Point2f>& pt_set1,
													                       const std::vector<cv::Point2f>& pt_set2,
													                       const cv::Mat& K,
													                       const cv::Mat& Kinv,
													                       const cv::Mat& P1,
													                       const cv::Mat& P2,
													                       pcl::PointCloud<pcl::PointXYZRGB>& pointcloud)
{
	pointcloud.points.clear();

	pcl::PointXYZRGB point;

	for (unsigned int iii = 0; iii < pt_set1.size(); iii++) {    
    Triangulate(pt_set1.at(iii), pt_set2.at(iii), K, Kinv,  P1, P2, point, false);
    pointcloud.points.push_back(point);
  }
}

void sfm::StructureFromMotion::projectionToTransformation(const cv::Mat& P, cv::Mat& C)
{
	C = cv::Mat(4,4,P.type());
	for(int i=0; i<3; i++)
		for(int j=0; j<4; j++)
			C.at<double>(i,j) = P.at<double>(i,j);
	C.at<double>(3,3) = 1.0;
}

void sfm::StructureFromMotion::transformationToProjection(const cv::Mat& C, cv::Mat& P)
{
	P = cv::Mat(3,4,C.type());
	for(int i=0; i<3; i++)
		for(int j=0; j<4; j++)
			P.at<double>(i,j) = C.at<double>(i,j);
}

void sfm::StructureFromMotion::Triangulate(const cv::Point2f& pt1, 
																					 const cv::Point2f& pt2, 
																					 const cv::Mat& K, 
																					 const cv::Mat& Kinv, 
																					 const cv::Mat& P1, 
																					 const cv::Mat& P2, 
																					 pcl::PointXYZRGB& xyzPoint, 
																					 bool debug) 
{
	cv::Mat C1, C2;
	projectionToTransformation(P1, C1);
	projectionToTransformation(P2, C2);

	// Relative
	cv::Mat P0, PR, C0, CR;
	//initializeP0(P0);
	P0 = (cv::Mat_<double>(3,4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
	projectionToTransformation(P0, C0);
	CR = C2 * C1.inv();
	transformationToProjection(CR, PR);

	// suposa que P1 i P2 són les matrius de proj en coordenades
	// absolutes i, per tant, transforma a relatives amb
	// P0 = Id + columna de zeros
	// PR = relativa de P2 a P1

	cv::Point3d xyzPoint_R;

	//cv::Point2f kp = pt1;
	cv::Point3d u_1(pt1.x, pt1.y, 1.0);
	cv::Mat umx1 = Kinv * cv::Mat(u_1);


	u_1.x = umx1.at<double>(0, 0);
	u_1.y = umx1.at<double>(1, 0);
	u_1.z = umx1.at<double>(2, 0);

	//cv::Point2f kp1 = pt2;
	cv::Point3d u_2(pt2.x, pt2.y,1.0);
	cv::Mat umx2 = Kinv * cv::Mat(u_2);

	u_2.x = umx2.at<double>(0, 0);
	u_2.y = umx2.at<double>(1, 0);
	u_2.z = umx2.at<double>(2, 0);

	cv::Mat X;

	IterativeLinearLSTriangulation(X, u_1, P0, u_2, PR);

	xyzPoint_R = cv::Point3d( X.at<double>(0, 0), X.at<double>(1, 0), X.at<double>(2, 0) );

	cv::Mat A(4, 1, CV_64FC1), B(4, 1, CV_64FC1);

	A.at<double>(0,0) = xyzPoint_R.x;
	A.at<double>(1,0) = xyzPoint_R.y;
	A.at<double>(2,0) = xyzPoint_R.z;
	A.at<double>(3,0) = 1.0;

	B = C1 * A;

	xyzPoint.x = B.at<double>(0,0) / B.at<double>(3,0);
	xyzPoint.y = B.at<double>(1,0) / B.at<double>(3,0);
	xyzPoint.z = B.at<double>(2,0) / B.at<double>(3,0);

  cv::Scalar color = curr_img_.at<unsigned char>(pt2);

  xyzPoint.r = color[0];
  xyzPoint.g = color[1];
  xyzPoint.b = color[2];

	if (debug) {
	       
		printf("%s << DEBUG SUMMARY:\n", __FUNCTION__);


		std::cout << std::endl << "P1 = " << P1 << std::endl;
		std::cout << "P2 = " << P2 << std::endl;
		std::cout << "P0 = " << P0 << std::endl;
		std::cout << "PR = " << PR << std::endl;
		std::cout << "C1 = " << C1 << std::endl << std::endl;

		printf("pt1 -> u_1 = (%f, %f), (%f, %f, %f)\n", pt1.x, pt1.y, u_1.x, u_1.y, u_1.z);
		printf("pt2 -> u_2 = (%f, %f), (%f, %f, %f)\n\n", pt2.x, pt2.y, u_2.x, u_2.y, u_2.z);

		printf("xyzPoint_R = (%f, %f, %f)\n", xyzPoint_R.x, xyzPoint_R.y, xyzPoint_R.z);
		printf("xyzPoint = (%f, %f, %f)\n\n", xyzPoint.x, xyzPoint.y, xyzPoint.z);

		std::cin.get();
	}
         
 
}

void sfm::StructureFromMotion::IterativeLinearLSTriangulation(      cv::Mat&    dst, 
																															const cv::Point3d& u1, 
																															const cv::Mat&     P1, 
																															const cv::Point3d& u2, 
																															const cv::Mat&     P2) 
{
         
	int maxIterations = 10; //Hartley suggests 10 iterations at most

	cv::Mat X(4, 1, CV_64FC1), XA;

	LinearLSTriangulation(XA, u1, P1, u2, P2);

	X.at<double>(0,0) = XA.at<double>(0,0);
	X.at<double>(1,0) = XA.at<double>(1,0); 
	X.at<double>(2,0) = XA.at<double>(2,0); 
	X.at<double>(3,0) = 1.0;

	double wi1 = 1.0, wi2 = 1.0;

	for (int i = 0; i < maxIterations; i++) {
		// recalculate weights
		cv::Mat P1X = P1.row(2) * X;
		double p1a = P1X.at<double>(0, 0);
		cv::Mat P2X = P2.row(2) * X;
		double p2a = P2X.at<double>(0, 0);
		// breaking point
		if ((fabsf(wi1 - p1a) <= EPSILON) && (fabsf(wi2 - p2a) <= EPSILON)) 
		{
			break;
		} 

		wi1 = p1a;
		wi2 = p2a;

		// reweight equations and solve
		cv::Mat A(4, 3, CV_64FC1);
		
		A.at<double>(0,0) = (u1.x * P1.at<double>(2,0) - P1.at<double>(0,0)) / wi1;
		A.at<double>(0,1) = (u1.x * P1.at<double>(2,1) - P1.at<double>(0,1)) / wi1;
		A.at<double>(0,2) = (u1.x * P1.at<double>(2,2) - P1.at<double>(0,2)) / wi1;

		A.at<double>(1,0) = (u1.y * P1.at<double>(2,0) - P1.at<double>(1,0)) / wi1;
		A.at<double>(1,1) = (u1.y * P1.at<double>(2,1) - P1.at<double>(1,1)) / wi1;
		A.at<double>(1,2) = (u1.y * P1.at<double>(2,2) - P1.at<double>(1,2)) / wi1;

		A.at<double>(2,0) = (u2.x * P2.at<double>(2,0) - P2.at<double>(0,0)) / wi2;
		A.at<double>(2,1) = (u2.x * P2.at<double>(2,1) - P2.at<double>(0,1)) / wi2;
		A.at<double>(2,2) = (u2.x * P2.at<double>(2,2) - P2.at<double>(0,2)) / wi2;

		A.at<double>(3,0) = (u2.y * P2.at<double>(2,0) - P2.at<double>(1,0)) / wi2;
		A.at<double>(3,1) = (u2.y * P2.at<double>(2,1) - P2.at<double>(1,1)) / wi2;
		A.at<double>(3,2) = (u2.y * P2.at<double>(2,2) - P2.at<double>(1,2)) / wi2;

		cv::Mat B(4, 1, CV_64FC1);

		B.at<double>(0,0) = -(u1.x * P1.at<double>(2,3) - P1.at<double>(0,3)) / wi1;
		B.at<double>(1,0) = -(u1.y * P1.at<double>(2,3) - P1.at<double>(1,3)) / wi1;
		B.at<double>(2,0) = -(u2.x * P2.at<double>(2,3) - P2.at<double>(0,3)) / wi2;
		B.at<double>(3,0) = -(u2.y * P2.at<double>(2,3) - P2.at<double>(1,3)) / wi2;

		cv::solve(A, B, XA, cv::DECOMP_SVD);

		X.at<double>(0,0) = XA.at<double>(0,0);
		X.at<double>(1,0) = XA.at<double>(1,0); 
		X.at<double>(2,0) = XA.at<double>(2,0); 
		X.at<double>(3,0) = 1.0;   
	}

	X.copyTo(dst);

 }

void sfm::StructureFromMotion::LinearLSTriangulation(      cv::Mat&    dst, // dst  : 3D point (homogeneous?)
																							 			 const cv::Point3d& u1, // u1   : image 1 homogenous 2D point
																						 				 const cv::Mat&     P1, // P1   : image 1 camera projection (3,4)
																				  	 				 const cv::Point3d& u2, // u2   : image 2 homogenous 2D point
																				 						 const cv::Mat&     P2) // P2   : image 2 camera projection     
{

	// https://github.com/MasteringOpenCV/code/blob/master/Chapter4_StructureFromMotion/Triangulation.cpp
	// http://www.morethantechnical.com/2012/01/04/simple-triangulation-with-opencv-from-harley-zisserman-w-code/

	// First, build matrix A for homogenous equation system Ax = 0

	cv::Mat A(4, 3, CV_64FC1);  
	A.at<double>(0,0) = u1.x * P1.at<double>(2,0) - P1.at<double>(0,0);
	A.at<double>(0,1) = u1.x * P1.at<double>(2,1) - P1.at<double>(0,1);
	A.at<double>(0,2) = u1.x * P1.at<double>(2,2) - P1.at<double>(0,2);

	A.at<double>(1,0) = u1.y * P1.at<double>(2,0) - P1.at<double>(1,0);
	A.at<double>(1,1) = u1.y * P1.at<double>(2,1) - P1.at<double>(1,1);
	A.at<double>(1,2) = u1.y * P1.at<double>(2,2) - P1.at<double>(1,2);

	A.at<double>(2,0) = u2.x * P2.at<double>(2,0) - P2.at<double>(0,0);
	A.at<double>(2,1) = u2.x * P2.at<double>(2,1) - P2.at<double>(0,1);
	A.at<double>(2,2) = u2.x * P2.at<double>(2,2) - P2.at<double>(0,2);

	A.at<double>(3,0) = u2.y * P2.at<double>(2,0) - P2.at<double>(1,0);
	A.at<double>(3,1) = u2.y * P2.at<double>(2,1) - P2.at<double>(1,1);
	A.at<double>(3,2) = u2.y * P2.at<double>(2,2) - P2.at<double>(1,2);

	// Assume X = (x,y,z,1), for Linear-LS method
	// Which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1

	cv::Mat B(4, 1, CV_64FC1);

	B.at<double>(0,0) = -((double) u1.x * P1.at<double>(2,3) - P1.at<double>(0,3));
	B.at<double>(0,1) = -((double) u1.y * P1.at<double>(2,3) - P1.at<double>(1,3));
	B.at<double>(0,2) = -((double) u2.x * P2.at<double>(2,3) - P2.at<double>(0,3));
	B.at<double>(0,3) = -((double) u2.y * P2.at<double>(2,3) - P2.at<double>(1,3));

	cv::solve(A, B, dst, cv::DECOMP_SVD);
}