//
// Created by pkyle on 1/20/25.
// Ultradent Products Inc.
// Copyright (c) 2025, Ultradent Products Inc. All rights reserved.
//
#ifndef FACE_POSE_PROJECT_POSE_ESTIMATION_BASE_FACE_POSE_H_
#define FACE_POSE_PROJECT_POSE_ESTIMATION_BASE_FACE_POSE_H_
#include "opencv2/opencv.hpp"

namespace re::face_pose::pose {
struct Face{
  Face() :x(0.0), y(0.0), z(0.0), confidence(0.0){}
  float x,y,z;
  double confidence;
};
class BaseFacePose{
 public:
  virtual ~BaseFacePose() = default;
  virtual bool Init(cv::Size image_size) = 0;
  virtual cv::Mat LookForFaces(const cv::Mat & frame, const double &min_confidence) = 0;
};
}  // namespace re::face_pose::pose


#endif //FACE_POSE_PROJECT_POSE_ESTIMATION_BASE_FACE_POSE_H_
