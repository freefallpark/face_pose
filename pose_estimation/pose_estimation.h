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
  class PriorBox{
   public:
    PriorBox();
    cv::Mat Decode( const cv::Mat& loc, const cv::Mat& conf, const cv::Mat& iou, float conf_threshold, int topK);
   private:
    void BuildPriors();

    cv::Size input_shape;
    cv::Size output_shape;
    std::vector<int> strides;
    std::vector<std::vector<float>> min_sizes;
    std::vector<cv::Vec4f> priors;
  };
  void Init();
  void DisplayVideo();

  // General
  std::atomic<bool> stop_;
  std::thread pose_thread_;

  // Oak D S2 general connection
  dai::Pipeline pipeline_;
  std::shared_ptr<dai::Device> device_;

  // Color Camera
  std::shared_ptr<dai::node::ColorCamera> cam_rgb_;
  std::shared_ptr<dai::node::XLinkOut>    xout_rgb_;
  std::shared_ptr<dai::DataOutputQueue>   q_rgb_;

  // Image Manip (need to resize and reformat color camera image to NN's Expected size)
  std::shared_ptr<dai::node::ImageManip> manip_;


  // Neural Network
  std::shared_ptr<dai::node::NeuralNetwork> nn_;
  std::shared_ptr<dai::node::XLinkOut> xout_nn_;
  std::shared_ptr<dai::DataOutputQueue>   q_det_;
};

} // re

#endif //FACE_POSE_POSE_ESTIMATION_POSE_ESTIMATION_H_
