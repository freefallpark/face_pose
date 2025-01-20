//
// Created by pkyle on 1/20/25.
//

#include "luxonis_camera.h"
#include <glog/logging.h>

namespace re {
namespace camera {
LuxonisCamera::LuxonisCamera() :device_(nullptr), cam_(nullptr), xout_(nullptr), q_cam_(nullptr){}
bool LuxonisCamera::Connect(const CamSettings& settings){
  std::lock_guard lock(mtx_);
  // Create Camera Node and set it up:
  cam_ = pipeline_.create<dai::node::ColorCamera>();
  cam_->setBoardSocket(dai::CameraBoardSocket::CAM_A);
  cam_->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
  cam_->setInterleaved(false);

  // user settings
  cam_->setPreviewSize(settings.frame_width, settings.frame_height);
  cam_->setFps(60);

  // Create outputs
  xout_ = pipeline_.create<dai::node::XLinkOut>();
  xout_->setStreamName("rgb");

  // Link Camera Output to manip and xout
  cam_->preview.link(xout_->input);

  try{
    // Connect to device
    device_ = std::make_shared<dai::Device>(pipeline_);
  }
  catch(std::runtime_error &e){
    LOG(ERROR) << "Failed to Connect to Camera: " << e.what();
    return false;
  }
  LOG(INFO) << "Usb Speed: " << device_->getUsbSpeed();

  // Bootloader Version
  if(device_->getBootloaderVersion()){
    LOG(INFO) << "Bootloader Version: " << device_->getBootloaderVersion()->toString();
  }

  // Device Name
  LOG(INFO) << "Device Name: " << device_->getDeviceName() << " Product Name: " << device_->getProductName();

  // Output Queue, used to get rgb frames
  q_cam_ = device_->getOutputQueue("rgb", 4, false);

  LOG(INFO) << "Camera Ready for Frame capture";

  return true;
}
cv::Mat LuxonisCamera::GetFrame(){
  std::lock_guard lock(mtx_);
  return q_cam_ ? q_cam_->get<dai::ImgFrame>()->getCvFrame() : cv::Mat();
}
} // camera
} // re