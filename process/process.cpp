//
// Created by pkyle on 12/26/24.
//

#include "process.h"

#include <glog/logging.h>

#include "camera/luxonis_camera.h"
#include "pose_estimation/opencv_face_pose.h"


namespace re::face_pose {
Process::Process() : stop_(false),
                     camera_(std::make_unique<camera::LuxonisCamera>()),
                     face_pose_estimator_(std::make_unique<pose::OpenCVFacePose>()){}
Process::~Process(){
  stop_ = true;
}
int Process::Run() {
  // Initialize
  camera::CamSettings settings;
  settings.frame_width = 720;
  settings.frame_height = 480;
  if( ! camera_->Connect(settings)) {
    LOG(ERROR) << "Failed To Initialized Camera";
    Shutdown();
    return 1;
  }
  if( ! face_pose_estimator_->Init(camera_->GetFrame().size())) {
    LOG(ERROR) << "Failed To Initialized Face Pose Estimator";
    Shutdown();
    return 1;
  }

  // Main Loop
  while(!stop_){
    // Get Frame
    auto frame = camera_->GetFrame();

    // Look For Faces
    auto faces = face_pose_estimator_->LookForFaces(frame, 0.75);

    //Draw Faces
    DrawFaces(frame, faces);


    // Display Frame
    DisplayFrame("debug", frame);

  }

  // Shutdown
  Shutdown();

  return 0;
}
void Process::Shutdown() {
  cv::destroyAllWindows();
  stop_ = true;
}
void Process::DisplayFrame( const std::string &name, const cv::Mat &frame) {
  cv::namedWindow(name);
  cv::imshow(name, frame);
  auto key = cv::waitKey(1);
  if(key == 'q'){
    stop_ = true;
  }
}

void Process::DrawFaces(const cv::Mat &frame, const cv::Mat &faces) {
  for(int i = 0; i < faces.rows; i++){
    // Draw bounding box
    cv::rectangle(frame, cv::Rect2i(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)), int(faces.at<float>(i, 2)), int(faces.at<float>(i, 3))), cv::Scalar(0, 255, 0));
    cv::circle(frame, cv::Point2i(int(faces.at<float>(i, 4)),  int(faces.at<float>(i, 5))),  2, cv::Scalar(255,   0,   0));
    cv::circle(frame, cv::Point2i(int(faces.at<float>(i, 6)),  int(faces.at<float>(i, 7))),  2, cv::Scalar(  0,   0, 255));
    cv::circle(frame, cv::Point2i(int(faces.at<float>(i, 8)),  int(faces.at<float>(i, 9))),  2, cv::Scalar(  0, 255,   0));
    cv::circle(frame, cv::Point2i(int(faces.at<float>(i, 10)), int(faces.at<float>(i, 11))), 2, cv::Scalar(255,   0, 255));
    cv::circle(frame, cv::Point2i(int(faces.at<float>(i, 12)), int(faces.at<float>(i, 13))), 2, cv::Scalar(  0, 255, 255));
    // Put score
    cv::putText(frame, cv::format("%.4f", faces.at<float>(i, 14)), cv::Point2i(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1))+15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
  }
}


}  // namespace re::face_pose