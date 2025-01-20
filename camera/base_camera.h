//
// Created by pkyle on 1/20/25.
//

#ifndef FACE_POSE_PROJECT_CAMERA_BASE_CAMERA_H_
#define FACE_POSE_PROJECT_CAMERA_BASE_CAMERA_H_

#include <opencv2/opencv.hpp>

namespace re {
namespace camera {
struct CamSettings{
  CamSettings() : frame_width(1920), frame_height(1080){}
  int frame_width;
  int frame_height;

};
class BaseCamera {
 public:
  BaseCamera()  = default;
  ~BaseCamera() = default;
  /**
   * @brief Connects to camera, returns true upon success
   */
  virtual bool Connect(const CamSettings& settings) = 0;
  /**
   * @brief blocks until frame
   * @return cv::Mat containing frame data (empty if failed)
   */
  virtual cv::Mat GetFrame() = 0;
};

} // camera
} // re

#endif //FACE_POSE_PROJECT_CAMERA_BASE_CAMERA_H_
