//
// Created by pkyle on 12/26/24.
//

#include <csignal>
#include <string>

#include "glog/logging.h"

#include "process/process.h"

namespace {
volatile sig_atomic_t stop = 0;
void IntHandler(int signum){
  stop = signum;
}
}  // namespace

int main(int argc, char* argv[]){
  //Setup GLog
  std::string log_name = "nv_pose";
  if(argc > 1){
    log_name = argv[1];
  }
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(log_name.c_str());

  // Create Instance of Face Pose Process
  re::face_pose::Process process;

  // Run Process
  return process.Run();
}