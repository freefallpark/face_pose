//
// Created by pkyle on 12/26/24.
//
//TODO Clean this shit up, and make sure you undestand everything

#ifndef FACE_POSE_POSE_ESTIMATION_POSE_ESTIMATION_H_
#define FACE_POSE_POSE_ESTIMATION_POSE_ESTIMATION_H_

#include <atomic>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "base_face_pose.h"

namespace re::face_pose::pose {

class OpenCVFacePose final : public BaseFacePose{
 public:
  OpenCVFacePose() = default;
  OpenCVFacePose(OpenCVFacePose&) = delete;
  OpenCVFacePose& operator=(OpenCVFacePose&) = delete;
  ~OpenCVFacePose() override = default;

  bool Init(cv::Size image_size) override;
  cv::Mat LookForFaces(const cv::Mat &frame, const double &min_confidence) override;

 private:
  const std::string model_path_ = "/home/pkyle/reflective_encounters/face_pose/pose_estimation"
                                  "/face_detection_yunet_2023mar.onnx";
  cv::Ptr<cv::FaceDetectorYN> detector_;

};

}  // namespace re::face_pose::pose

#endif //FACE_POSE_POSE_ESTIMATION_POSE_ESTIMATION_H_
