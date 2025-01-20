//
// Created by pkyle on 12/26/24.
//

#include "process.h"

#include "glog/logging.h"

namespace re::face_pose {

Process::~Process() {
  if(pose_thread_.joinable()){
    pose_thread_.join();
  }
}
bool Process::Start() {
  if(! pose_estimation_.Init()){
    LOG(INFO) << "Failed to Initialize Face Pose Estimation";
    return false;
  }
  pose_thread_ = std::thread(&PoseEstimation::Run, &pose_estimation_);
  return true;
}
void Process::Stop() {
  // Shutdown pose
  pose_estimation_.Stop();
}


}  // namespace re::face_pose