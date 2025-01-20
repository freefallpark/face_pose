//
// Created by pkyle on 12/26/24.
//
//TODO Clean this shit up, and make sure you undestand everything

#ifndef FACE_POSE_POSE_ESTIMATION_POSE_ESTIMATION_H_
#define FACE_POSE_POSE_ESTIMATION_POSE_ESTIMATION_H_

#include <atomic>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "camera/luxonis_camera.h"
#include "depthai/depthai.hpp"

namespace re {

struct Face{
  Face():x(0),y(0),z(0),confidence(0){}
  float x,y,z;
  double confidence;
};

class PoseEstimation {
 public:
  PoseEstimation();
  PoseEstimation(PoseEstimation&) = delete;
  PoseEstimation& operator=(PoseEstimation&) = delete;
  ~PoseEstimation();

  bool Init();
  bool Run();
  void Stop();

 private:
  void DisplayFrame( const std::string &name, const cv::Mat &frame);

  std::vector<Face> LookForFaces(const cv::Mat &frame);

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

  // Camera
  std::unique_ptr<re::camera::BaseCamera> camera_;

  // Neural Network

};

} // re

#endif //FACE_POSE_POSE_ESTIMATION_POSE_ESTIMATION_H_
