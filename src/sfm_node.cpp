#include <ros/ros.h>
#include <sfm/sfm.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "laser_calibration_node");

  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  if (argc < 3){
    std::cout << "\tUsage: " << argv[0] << " output.jpg input1.jpg input2.jpg... " << std::endl;
    return -1;
  }
    
  int num_imgs = argc - 2;
  std::string output = argv[1];
  std::vector<cv::Mat> imgs(num_imgs);

  ROS_INFO_STREAM("[SFM] Input: " << num_imgs << " images.");

  //try to open the images:
  for(int i=0; i<num_imgs; i++)
  {
  	ROS_INFO_STREAM("[SFM] Opening image " << argv[2+i]);
    try
    {
      imgs[i]=cv::imread(argv[2+i]);
      if( !imgs[i].data )
         throw "Could not read image";
      //ROS_INFO_STREAM("[SFM] Image is " << imgs[i].cols << "x" << imgs[i].rows);
    }
    catch( char * str )
    {
      ROS_ERROR_STREAM("Exception raised: " << str  << argv[2+i]);
    }
  }

  sfm::StructureFromMotion sfm(imgs);

  return 0;
}