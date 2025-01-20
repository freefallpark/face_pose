//
// Created by pkyle on 1/20/25.
//

#ifndef FACE_POSE_PROJECT_CAMERA_LUXONIS_CAMERA_H_
#define FACE_POSE_PROJECT_CAMERA_LUXONIS_CAMERA_H_


#include "base_camera.h"
#include "depthai/depthai.hpp"

namespace re {
namespace camera {

class LuxonisCamera final : public BaseCamera{
 public:
  LuxonisCamera();
  /**
   * @brief Connects to camera, returns true upon success
   */
  bool Connect(const CamSettings& settings) override;
  /**
   * @brief blocks until frame
   * @return cv::Mat containing frame data (empty if failed)
   */
  cv::Mat GetFrame() override;

 private:
  std::mutex mtx_;
  // Oak D S2 general connection
  dai::Pipeline pipeline_;
  std::shared_ptr<dai::Device> device_;

  // Color Camera
  std::shared_ptr<dai::node::ColorCamera>  cam_;
  std::shared_ptr<dai::node::XLinkOut>    xout_;
  std::shared_ptr<dai::DataOutputQueue>   q_cam_;


};

} // camera
} // re

#endif //FACE_POSE_PROJECT_CAMERA_LUXONIS_CAMERA_H_
