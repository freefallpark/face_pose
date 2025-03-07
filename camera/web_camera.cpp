//
// Created by pkyle on 3/6/25.
// Ultradent Products Inc.
// Copyright (c) 2025, Ultradent Products Inc. All rights reserved.
//
#include "web_camera.h"

namespace camera {
} // camera
bool re::camera::WebCamera::Connect(const re::camera::CamSettings &settings) {
  if( !cap_.open(0)){
    return false;
  }
  frame_size_ = cv::Size(settings.frame_width,settings.frame_height);
  return true;
}

cv::Mat re::camera::WebCamera::GetFrame() {
  cv::Mat frame;
  cap_.read(frame);
  cv::resize(frame,frame, frame_size_);
  return frame;
}
