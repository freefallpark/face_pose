//
// Created by pkyle on 12/26/24.
//

#include "pose_estimation.h"
#include "opencv2/highgui.hpp"

#include "glog/logging.h"

namespace re {
static constexpr int kNnWidth = 160;
static constexpr int kNnHeight = 120;
static constexpr int kRgbWidth = 640;
static constexpr int kRgbHeight = 480;


PoseEstimation::PoseEstimation() : stop_(false), pipeline_(), device_(){}
PoseEstimation::~PoseEstimation() {
  Stop();
  if(pose_thread_.joinable()){
    pose_thread_.join();
  }
}
void PoseEstimation::Init(){
  // Create Mono Camera Node:
  cam_mono_ = pipeline_.create<dai::node::MonoCamera>();
  cam_mono_->setBoardSocket(dai::CameraBoardSocket::CAM_B);
  cam_mono_->setResolution(dai::MonoCameraProperties::SensorResolution::THE_720_P);

  // Create Mono Image Manip Node:
  manip_mono_ = pipeline_.create<dai::node::ImageManip>();
  manip_mono_->initialConfig.setResize(kRgbWidth, kRgbHeight);
  manip_mono_->setKeepAspectRatio(false);

  // Link mon image to manip input
  cam_mono_->out.link(manip_mono_->inputImage);

  // Crate xlink out to send resized frames to host
  xout_mono_ = pipeline_.create<dai::node::XLinkOut>();
  xout_mono_->setStreamName("mono");
  manip_mono_->out.link(xout_mono_->input);

  // Create NN Image Manip Node:
  manip_nn_ = pipeline_.create<dai::node::ImageManip>();
  manip_nn_->initialConfig.setResize(kNnWidth, kNnHeight);
  manip_nn_->initialConfig.setFrameType(dai::RawImgFrame::Type::BGR888p);
  manip_nn_->setKeepAspectRatio(false);

  // Link Camera Output to ImageManip input
  cam_mono_->out.link(manip_nn_->inputImage);

  // Create Neural Network Node
  nn_ = pipeline_.create<dai::node::NeuralNetwork>();
  dai::OpenVINO::Blob blob("/home/pkyle/reflective_encounters/face_pose/pose_estimation/face_detection_yunet_160x120.blob");
  nn_->setBlob(blob);
  nn_->setNumInferenceThreads(2);
  nn_->input.setBlocking(false);

  // Link manip outut to nn input
  manip_nn_->out.link(nn_->input);

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
  q_mono_ = device_->getOutputQueue("mono", 4, false);
  q_det_ = device_->getOutputQueue("nn", 4, false);

}
void PoseEstimation::DisplayVideo() {
  cv::namedWindow("rgb");
  while( ! stop_.load()){
    // Wait for Camera Frame
    auto in_rgb = q_mono_->get<dai::ImgFrame>();
    auto frame = in_rgb->getCvFrame();

    // Try to get NN output
    auto in_det = q_det_->tryGet<dai::NNData>();

    // Parse output
    std::vector<float> conf, iou, loc;
    cv::Mat conf_mat, iou_mat, loc_mat, detected_faces;
    if(in_det != nullptr){
      conf = in_det->getLayerFp16("conf");
      iou = in_det->getLayerFp16("iou");
      loc = in_det->getLayerFp16("loc");
      conf_mat = cv::Mat(1076,2, CV_32F, conf.data());
      iou_mat = cv::Mat(1076,1, CV_32F, iou.data());
      loc_mat = cv::Mat(1076,14, CV_32F, loc.data());
      // Get Confidence
      std::vector<Face> faces;
      for(int i = 0; i < 1076; i++){
        auto cls_score = conf_mat.at<float>(i,1);
        auto iou_score = iou_mat.at<float>(i,0);
        auto face_confidence = std::min(cls_score, iou_score);
        if(face_confidence > 0.75){
          Face face;
          face.confidence = face_confidence;

          //Solve for cx
          face.cx = loc_mat.at<float>(i,0);

          //Solve for cy
          face.cy = loc_mat.at<float>(i,1);

          faces.push_back(face);
        }
      }
      auto max_it = std::max_element(faces.begin(), faces.end(), [](const Face& a, const Face& b){
        return a.confidence < b.confidence;
      });
      if(max_it != faces.end()){
        std::cout << "\rMax Confidence: " << max_it->confidence
                  << " x: " << max_it->cx
                  << " y: " << max_it->cy << std::flush;
      }
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