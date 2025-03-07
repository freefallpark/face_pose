//
// Created by pkyle on 12/26/24.
//

#ifndef FACE_POSE_PROJECT_PROCESS_PROCESS_H_
#define FACE_POSE_PROJECT_PROCESS_PROCESS_H_

#include <atomic>

#include <opencv2/opencv.hpp>

#include "camera/base_camera.h"
#include "pose_estimation/base_face_pose.h"

namespace re::face_pose {

class Process {
 public:
  explicit Process(const std::string &model_path);
  ~Process();

  int Run();

 private:
  void Shutdown();
  static void DrawFaces( const cv::Mat& frame, const cv::Mat &faces);
  static void DrawFaceTargets( const cv::Mat &frame, const cv::Mat &faces);
  void DisplayFrame( const std::string &name, const cv::Mat& frame);
  // Process Related Members
  std::atomic<bool> stop_;
  // Cameras
  std::unique_ptr<camera::BaseCamera> luxonis_camera_;
  std::unique_ptr<camera::BaseCamera> web_camera_;
  // Face Pose Estimator
  std::unique_ptr<pose::BaseFacePose> luxonis_estimator_;
  std::unique_ptr<pose::BaseFacePose> web_estimator_;
};

}  // namespace re::face_pose

#endif //FACE_POSE_PROJECT_PROCESS_PROCESS_H_
