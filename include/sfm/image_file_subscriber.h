#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_geometry/pinhole_camera_model.h>
#include <camera_calibration_parsers/parse_yml.h>

#include <boost/foreach.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/algorithm/string/predicate.hpp>

class Subscriber
{
public:
	Subscriber()
	{
		ros::NodeHandle nhp("~");

		std::string camera_info_yaml,camera_name;
	  nhp.param("camera_info_yaml", camera_info_yaml, std::string());
	  nhp.param("camera_name", camera_name, std::string());

		// Load the camera info yaml file
	  camera_calibration_parsers::readCalibrationYml(camera_info_yaml, camera_name, cam_info_);
	  ROS_INFO_STREAM("[Subscriber]: Camera info YAML file loaded correctly.");
	}

	typedef boost::function<void (const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&)> CallbackType;

  void registerCallback(const CallbackType& callback)
  {
    callback_ = callback;
  }

  image_geometry::PinholeCameraModel getCameraModel(void)
  {
  	image_geometry::PinholeCameraModel pcm;
  	pcm.fromCameraInfo(cam_info_);
  	return pcm;
  }

  virtual void subscribe();

private:
	CallbackType callback_;
	sensor_msgs::CameraInfo cam_info_;

	void triggerCallback(const sensor_msgs::ImageConstPtr& image)
  {
  	sensor_msgs::CameraInfoConstPtr c(cam_info_);
  	callback_(image,c);
  }

  void triggerCallback(const sensor_msgs::ImageConstPtr& image, const sensor_msgs::CameraInfoConstPtr& c)
  {
  	callback_(image,c);
  }

};

class ImageFileSubscriber : Subscriber
{
public:
	ImageFileSubscriber(const std::string& path, 
											const std::string& pattern,
											const std::string& extension):
											path_(path),pattern_(pattern),extension_(extension)
	{

		namespace fs = boost::filesystem;

    fs::path full_path = fs::system_complete( fs::path( path ) );

    if ( !fs::exists( full_path ) )
	  {
	    //ROS_ERROR_STREAM("[ImageFileSubscriber]: Folder " << full_path.file_string() << " not found");
	    ROS_ERROR_STREAM("[ImageFileSubscriber]: Folder not found");
	    return 1;
	  }
	  if ( fs::is_directory( full_path ) )
	  {
      //ROS_INFO_STREAM("[ImageFileSubscriber]: In directory " << full_path.directory_string());
      ROS_INFO_STREAM("[ImageFileSubscriber]: In directory ");

	    fs::directory_iterator end_iter;

	    unsigned long image_count = 0;

	    for ( fs::directory_iterator dir_itr( full_path );
	          dir_itr != end_iter;
	          ++dir_itr )
	    {
	      try
	      {
	        if ( fs::is_regular_file( dir_itr->status() ) )
	        {
	        	if (boost::starts_with(dir_itr->path().filename(), pattern_))
        		{
        			std::string image = dir_itr->path().filename();
        			image_names_.push_back(image);
        			++image_count;
        			ROS_INFO_STREAM("[ImageFileSubscriber]: Found " << image.c_str() << " image.");
        		}

	        }
	      }
	      catch ( const std::exception & ex )
	      {
	        ++err_count;
	        std::cout << dir_itr->path().filename() << " " << ex.what() << std::endl;
	      }
	    }
	    ROS_INFO_STREAM("[ImageFileSubscriber]: " << image_count << " files");
	  }else{
	  	ROS_ERROR_STREAM("[ImageFileSubscriber]: Path " << full_path.file_string() << " is not a folder");
	  	return 1;
	  }

	  current_idx_ = 0;

	}

	void readImage(cv::Mat& img)
	{
		if(current_idx_ < image_names_.size())
		{
			img = cv::imread(path_ + image_names_[current_idx_]);
			current_idx_++;
		}
	}

	void subscribe()
	{
		if(image_names_.size()>0)
		{
			for(size_t i=0; i<image_names_.size(); i++)
			{
				cv::Mat img = cv::imread(path_ + image_names_[i]);
				std_msgs::Header header;
				header.stamp = ros::Time::now();
				cv_bridge::CvImage cv_img(header,encoding,img);
				sensor_msgs::Image image_msg;
				cv_img.toImageMsg(image_msg);
				triggerCallback(image_msg);
			}
		}else{
			ROS_ERROR_STREAM("No images were found. Exitting.");
			return;
		}
	}

private:
	std::string path_;
	std::string pattern_;
	std::string extension_;
	std::vector<std::string> image_names_;
	int current_idx_;

};