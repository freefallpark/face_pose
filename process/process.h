//
// Created by pkyle on 12/26/24.
//

#ifndef FACE_POSE_PROJECT_PROCESS_PROCESS_H_
#define FACE_POSE_PROJECT_PROCESS_PROCESS_H_

#include "pose_estimation/pose_estimation.h"

namespace re::face_pose {

class Process {
 public:
  bool Run();
  void Stop();

 private:
  PoseEstimation pose_estimation_;
};

}  // namespace re::face_pose

#endif //FACE_POSE_PROJECT_PROCESS_PROCESS_H_
