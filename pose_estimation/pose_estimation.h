//
// Created by pkyle on 12/26/24.
//

#ifndef FACE_POSE_POSE_ESTIMATION_POSE_ESTIMATION_H_
#define FACE_POSE_POSE_ESTIMATION_POSE_ESTIMATION_H_

#include <atomic>

#include "depthai/depthai.hpp"

namespace re {


class PoseEstimation {
 public:
  PoseEstimation();
  PoseEstimation(PoseEstimation&) = delete;
  PoseEstimation& operator=(PoseEstimation&) = delete;
  ~PoseEstimation();

  bool Run();
  void Stop();

 private:
  struct Face{
    float confidence = 0;
    float cx = 0;
    float cy = 0;
  };
  void Init();
  void DisplayVideo();
  /**
    * @brief Draw bounding boxes and landmarks on an image.
    *
    * The detections Mat should have shape (N x 15) with each row:
    *   [x1, y1, x2, y2, lmk1_x, lmk1_y, lmk2_x, lmk2_y, lmk3_x, lmk3_y,
    *    lmk4_x, lmk4_y, lmk5_x, lmk5_y, score]
    *
    * @param frame      The original image on which to draw.
    * @param detections The (N x 15) Mat of detections.
    * @return           A copy of the frame with bounding boxes and landmarks drawn.
    */
  void DrawFaces(cv::Mat& frame, const cv::Mat& detections);

  // General
  std::atomic<bool> stop_;
  std::thread pose_thread_;

  // Oak D S2 general connection
  dai::Pipeline pipeline_;
  std::shared_ptr<dai::Device> device_;

  // Color Camera
  std::shared_ptr<dai::node::ColorCamera>  cam_;
  std::shared_ptr<dai::node::XLinkOut>    xout_;
  std::shared_ptr<dai::DataOutputQueue>   q_cam_;

  // Image Manip (need to resize and reformat color camera image to NN's Expected size)
  std::shared_ptr<dai::node::ImageManip> manip_nn_;
  std::shared_ptr<dai::node::ImageManip> manip_;

  // Neural Network
  std::shared_ptr<dai::node::NeuralNetwork> nn_;
  std::shared_ptr<dai::node::XLinkOut>      xout_nn_;
  std::shared_ptr<dai::DataOutputQueue>     q_nn_;
};

} // re

#endif //FACE_POSE_POSE_ESTIMATION_POSE_ESTIMATION_H_
