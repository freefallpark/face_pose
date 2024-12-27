//
// Created by pkyle on 12/26/24.
//

#include "pose_estimation.h"
#include "opencv2/highgui.hpp"

#include "glog/logging.h"

namespace re {

PoseEstimation::PoseEstimation() : stop_(false), pipeline_(), device_(){}
PoseEstimation::~PoseEstimation() {
  Stop();
  if(pose_thread_.joinable()){
    pose_thread_.join();
  }
}
void PoseEstimation::Init(){
  // Create Camera and output nodes on pipeline
  cam_rgb_ = pipeline_.create<dai::node::ColorCamera>();
  xout_rgb_ = pipeline_.create<dai::node::XLinkOut>();

  // Setup
  xout_rgb_->setStreamName("rgb");
  cam_rgb_->setPreviewSize(640,480);
  cam_rgb_->setBoardSocket(dai::CameraBoardSocket::CAM_A);
  cam_rgb_->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
  cam_rgb_->setInterleaved(false);
  cam_rgb_->setColorOrder(dai::ColorCameraProperties::ColorOrder::RGB);

  // Linking
  cam_rgb_->preview.link(xout_rgb_->input);

  // Connect to device
  device_ = std::make_shared<dai::Device>(pipeline_, dai::UsbSpeed::SUPER);
  LOG(INFO) << "Connected cameras: " << device_->getConnectedCameraFeatures();
  LOG(INFO) << "Usb Speed: " << device_->getUsbSpeed();

  // Bootloader Version
  if(device_->getBootloaderVersion()){
    LOG(INFO) << "Bootloader Version: " << device_->getBootloaderVersion()->toString();
  }

  // Device Name
  LOG(INFO) << "Device Name: " << device_->getDeviceName() << " Product Name: " << device_->getProductName();

  // Output Queue, used to get rgb frames
  q_rgb_ = device_->getOutputQueue("rgb", 4, false);
}
void PoseEstimation::DisplayVideo() {
  cv::namedWindow("rgb");
  while( ! stop_.load()){
    auto in_rgb = q_rgb_->get<dai::ImgFrame>();
    cv::imshow("rgb", in_rgb->getCvFrame());
    cv::waitKey(1);
  }
  cv::destroyAllWindows();
}
bool PoseEstimation::Run() {
  // Initialize
  try{
    Init();
  }
  catch(...){
    Stop();
    return false;
  }

  // Start Video Display thread
  pose_thread_ = std::thread(&PoseEstimation::DisplayVideo, this);

  LOG(INFO) << "Successfully started Video Display";

  return true;
}
void PoseEstimation::Stop() {
  stop_.store(true);
}
} // re