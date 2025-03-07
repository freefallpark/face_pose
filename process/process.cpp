//
// Created by pkyle on 12/26/24.
//

#include "process.h"

#include <glog/logging.h>

#include "camera/luxonis_camera.h"
#include "pose_estimation/opencv_face_pose.h"
#include "camera/web_camera.h"


namespace re::face_pose {
Process::Process(const std::string &model_path)
    : stop_(false),
      luxonis_camera_(std::make_unique<camera::LuxonisCamera>()),
      web_camera_(std::make_unique<camera::WebCamera>()),
      luxonis_estimator_(std::make_unique<pose::OpenCVFacePose>(model_path)),
      web_estimator_(std::make_unique<pose::OpenCVFacePose>(model_path)){}
Process::~Process(){
  stop_ = true;
}
int Process::Run() {
  // Initialize Cameras
  camera::CamSettings settings;
  settings.frame_width = 640;
  settings.frame_height = 480;
  if( ! luxonis_camera_->Connect(settings)) {
    LOG(ERROR) << "Failed To Initialized Luxonis Camera";
    Shutdown();
    return 1;
  }
  if( ! web_camera_->Connect(settings)){
    LOG(ERROR) << "Failed To Initialized Web Camera";
    Shutdown();
    return 1;
  }

  // Initialize Estimators
  if( ! luxonis_estimator_->Init(luxonis_camera_->GetFrame().size())) {
    LOG(ERROR) << "Failed To Initialized Luxonis Estimator";
    Shutdown();
    return 1;
  }
  if( ! web_estimator_->Init(web_camera_->GetFrame().size())) {
    LOG(ERROR) << "Failed To Initialized Web Estimator";
    Shutdown();
    return 1;
  }

  // Main Loop
  while(!stop_){
    // Get Frames
    auto luxonis_frame = luxonis_camera_->GetFrame();
    auto web_frame = web_camera_->GetFrame();

    // Look For Faces
    auto luxonis_faces = luxonis_estimator_->LookForFaces(luxonis_frame, 0.75);
    auto web_faces = web_estimator_->LookForFaces(web_frame, 0.75);

    //Draw Faces
    DrawFaces(luxonis_frame, luxonis_faces);
    DrawFaces(web_frame, web_faces);

    // Draw Target
    DrawFaceTargets(luxonis_frame, luxonis_faces);
    DrawFaceTargets(web_frame, web_faces);

    // Display Frame
    DisplayFrame("luxonis", luxonis_frame);
    DisplayFrame("web", web_frame);
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
void Process::DrawFaceTargets([[maybe_unused]] const cv::Mat &frame,[[maybe_unused]] const cv::Mat &faces){
  //Target Right eye is at faces.at<float>(i, 4), left eye at (i,6)
  std::vector<cv::Point2f> targets;
  for(int i = 0; i< faces.rows; i++){
    //Left eye coords:
    cv::Point2f left_eye = cv::Point2f(faces.at<float>(i,4), faces.at<float>(i,5));
    cv::Point2f right_eye = cv::Point2f(faces.at<float>(i,6), faces.at<float>(i,7));
    cv::Point2f target = (left_eye + right_eye)*0.5f;
    cv::drawMarker(frame, target, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 10, 2);
    std::cout << "\r target: " << target << std::flush;
  }

}



}  // namespace re::face_pose