//
// Created by pkyle on 12/26/24.
//

#include "process.h"

#include "glog/logging.h"

namespace re::face_pose {

bool Process::Run() {
  if(! pose_estimation_.Run()){
    LOG(INFO) << "Failed during PoseEstimation::Run()";
    return false;
  }
  return true;
}
void Process::Stop() {
  pose_estimation_.Stop();

}
}  // namespace re::face_pose