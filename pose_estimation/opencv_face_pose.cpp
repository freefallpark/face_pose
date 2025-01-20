//
// Created by pkyle on 12/26/24.
//

#include "opencv_face_pose.h"

#include <utility>

#include "opencv2/highgui.hpp"
#include "prior_box.h"

#include "glog/logging.h"

namespace re::face_pose::pose {

bool face_pose::pose::OpenCVFacePose::Init(cv::Size image_size) {
  try{
    LOG(INFO) << "Setting up detector with model: " << model_path_;
    detector_ = cv::FaceDetectorYN::create(model_path_,"",cv::Size(640,640), 0.9,0.3,200);
    if( detector_.empty()) return false;
    detector_->setInputSize(image_size);
  }
  catch(...){
    LOG(ERROR) << "Failed to initialize Detector, Caught exception in init";
    return false;
  }
  return true;
}

cv::Mat face_pose::pose::OpenCVFacePose::LookForFaces(const cv::Mat &frame, const double &min_confidence) {
  cv::Mat faces_mat;
  try{
    detector_->detect(frame,faces_mat);
  }
  catch(...){
    LOG(ERROR) << "Failed to initialize Detector, Caught exception in LookForFaces";
    return {};
  }
  return faces_mat;
}

}  // namespace re::face_pose::pose