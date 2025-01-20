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

  cv::CommandLineParser parser(argc, argv,
                               "{model m|"
                               "/home/pkyle/reflective_encounters/face_pose/pose_estimation"
                               "/face_detection_yunet_2023mar.onnx"
                               "| Path to the model}");
  std::cout << "model path: " << parser.get<std::string>("model") << std::endl;
  //Setup GLog
  std::string log_name = "nv_pose";
  if(argc > 1){
    log_name = argv[1];
  }
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(log_name.c_str());

  // Create Instance of Face Pose Process
  re::face_pose::Process process(parser.get<std::string>("model"));

  // Run Process
  return process.Run();
}