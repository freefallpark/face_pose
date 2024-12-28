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
  // Create Color Camera Node:
  cam_rgb_ = pipeline_.create<dai::node::ColorCamera>();
  cam_rgb_->setPreviewSize(640,480);
  cam_rgb_->setBoardSocket(dai::CameraBoardSocket::CAM_A);
  cam_rgb_->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
  cam_rgb_->setInterleaved(false);

  // Create Color Camera Output:
  xout_rgb_ = pipeline_.create<dai::node::XLinkOut>();
  xout_rgb_->setStreamName("rgb");
  cam_rgb_->preview.link(xout_rgb_->input);


  // Create Image Manip Node:
  manip_ = pipeline_.create<dai::node::ImageManip>();
  manip_->initialConfig.setResize(160,120);
  manip_->initialConfig.setFrameType(dai::RawImgFrame::Type::BGR888p);
  manip_->setKeepAspectRatio(false);

  // Link Camera Output to ImageManip input
  cam_rgb_->preview.link(manip_->inputImage);

  // Create Neural Network Node
  nn_ = pipeline_.create<dai::node::NeuralNetwork>();
  dai::OpenVINO::Blob blob("/home/pkyle/reflective_encounters/face_pose/pose_estimation/face_detection_yunet_160x120.blob");
  nn_->setBlob(blob);
  nn_->setNumInferenceThreads(2);
  nn_->input.setBlocking(false);

  // Link manip outut to nn input
  manip_->out.link(nn_->input);

  // Create XLinkOut to get NN detections on the host
  xout_nn_ = pipeline_.create<dai::node::XLinkOut>();
  xout_nn_->setStreamName("nn");
  nn_->out.link(xout_nn_->input);

  // Connect to device
  device_ = std::make_shared<dai::Device>(pipeline_, dai::UsbSpeed::SUPER);
  LOG(INFO) << "Usb Speed: " << device_->getUsbSpeed();

  // Bootloader Version
  if(device_->getBootloaderVersion()){
    LOG(INFO) << "Bootloader Version: " << device_->getBootloaderVersion()->toString();
  }

  // Device Name
  LOG(INFO) << "Device Name: " << device_->getDeviceName() << " Product Name: " << device_->getProductName();

  // Output Queue, used to get rgb frames
  q_rgb_ = device_->getOutputQueue("rgb", 4, false);
  q_det_ = device_->getOutputQueue("nn", 4, false);

}
void PoseEstimation::DisplayVideo() {
  cv::namedWindow("rgb");
  while( ! stop_.load()){
    // Wait for Camera Frame
    auto in_rgb = q_rgb_->get<dai::ImgFrame>();
    auto frame = in_rgb->getCvFrame();

    // Try to get NN output
    auto in_det = q_det_->tryGet<dai::NNData>();

    // Parse output
    std::vector<float> conf, iou, loc;
    cv::Mat conf_mat, iou_mat, loc_mat;
    if(in_det != nullptr){
      conf = in_det->getLayerFp16("conf");
      iou = in_det->getLayerFp16("iou");
      loc = in_det->getLayerFp16("loc");
      conf_mat = cv::Mat(1076,2, CV_32F, conf.data());
      iou_mat = cv::Mat(1076,2, CV_32F, iou.data());
      loc_mat = cv::Mat(1076,2, CV_32F, loc.data());
      auto max_element = std::max_element(conf.begin(), conf.end());
      std::cout << "\r value of max conf value: " << *max_element << std::flush;
    }


    cv::imshow("rgb", frame);
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