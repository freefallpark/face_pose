//
// Created by pkyle on 3/6/25.
// Ultradent Products Inc.
// Copyright (c) 2025, Ultradent Products Inc. All rights reserved.
//
#ifndef FACE_POSE_PROJECT_CAMERA_WEB_CAMERA_H_
#define FACE_POSE_PROJECT_CAMERA_WEB_CAMERA_H_
#include "camera/base_camera.h"

namespace re::camera {

class WebCamera final : public BaseCamera{
 public:
  ~WebCamera() override = default;

  bool Connect(const CamSettings &settings) override;

  cv::Mat GetFrame() override;

 public:
 private:
  cv::VideoCapture cap_;
  cv::Size frame_size_;
};

} // camera

#endif //FACE_POSE_PROJECT_CAMERA_WEB_CAMERA_H_
