//
// Created by pkyle on 12/26/24.
//

#include "opencv_face_pose.h"

#include "opencv2/highgui.hpp"
#include "prior_box.h"

#include "glog/logging.h"

namespace re::face_pose::pose {

bool face_pose::pose::OpenCVFacePose::Init(cv::Size image_size) {
  detector_ = cv::FaceDetectorYN::create(model_path_,"",cv::Size(320,320), 0.9,0.3,200);
  if( detector_.empty()) return false;
  detector_->setInputSize(image_size);
  return true;
}

cv::Mat face_pose::pose::OpenCVFacePose::LookForFaces(const cv::Mat &frame, const double &min_confidence) {
  cv::Mat faces_mat;
  detector_->detect(frame,faces_mat);
  return faces_mat;
}

}  // namespace re::face_pose::pose