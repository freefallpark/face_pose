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

  // Setup Clean Exit
  signal(SIGINT, IntHandler);     // Ctl+c
  signal(SIGTERM, IntHandler);    // clion 'stop' button


  // Start Server
  re::face_pose::Process process;
  if(process.Init()){
    return 1;
  }

  LOG(INFO) << "Face pose estimation server is running";

  //Busy Loop
  while(!stop){
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  LOG(INFO) << "Face pose estimation server shutting down";

  return 0;
}