//
// Created by pkyle on 12/26/24.
//

#include "pose_estimation.h"

#include "opencv2/highgui.hpp"
#include "prior_box.h"

#include "glog/logging.h"

namespace re {
static constexpr int kNnWidth = 160;
static constexpr int kNnHeight = 120;
static constexpr int kRgbWidth = 640;
static constexpr int kRgbHeight = 480;


// Public
PoseEstimation::PoseEstimation() : stop_(false),
                                   camera_(std::make_unique<camera::LuxonisCamera>()){}
PoseEstimation::~PoseEstimation() {
  Stop();
}
bool PoseEstimation::Init(){
  //Setup Camera
  camera::CamSettings settings;
  if( ! camera_->Connect(settings)){
    LOG(ERROR) << "Failed to Connect to camera";
    return false;
  }

  // Setup Neural Network

  return true;
}
bool PoseEstimation::Run() {
  // Main loop
  while(!stop_){
    // Get Frame
    auto frame = camera_->GetFrame();
    if(frame.empty()){
      stop_ = true;
      return false;
    }

    // Run Frame through NN
    auto faces = LookForFaces(frame);

    //Display frame
    DisplayFrame("debug", frame);

  }

  // Shutdown
  cv::destroyAllWindows();
  return true;
}
void PoseEstimation::Stop() {
  stop_.store(true);
}

// Private
void PoseEstimation::DisplayFrame( const std::string &name, const cv::Mat &frame) {
  cv::namedWindow(name);
  cv::imshow(name,frame);
  auto key = cv::waitKey(1);
  if(key == 'q'){
    stop_ = true;
  }
}
std::vector<Face> PoseEstimation::LookForFaces(const cv::Mat &frame) {
  return std::vector<Face>();
}
void PoseEstimation::DrawFaces(cv::Mat &frame, const cv::Mat &faces) {
// --------------------------------------------------
  // 1) If no detections, do nothing
  // --------------------------------------------------
  if (faces.empty() || faces.rows == 0) {
    return;
  }

  // Example threshold values:
  float confidence_thresh = 0.5f;
  float iou_thresh        = 0.4f;
  int   keep_top_k        = 200;
  float fps               = 60.0f; // For demonstration

  // --------------------------------------------------
  // 2) Perform NMS with OpenCV's cv::dnn::NMSBoxes
  // --------------------------------------------------

  // Extract bounding boxes [x1,y1,x2,y2] and scores from 'faces'.
  std::vector<cv::Rect>  bboxes;
  std::vector<float>     scores;
  bboxes.reserve(faces.rows);
  scores.reserve(faces.rows);

  for (int i = 0; i < faces.rows; i++)
  {
    float x1 = faces.at<float>(i, 0);
    float y1 = faces.at<float>(i, 1);
    float x2 = faces.at<float>(i, 2);
    float y2 = faces.at<float>(i, 3);

    float score = faces.at<float>(i, 14);
    scores.push_back(score);

    // Convert [x1,y1,x2,y2] to cv::Rect
    float width  = x2 - x1;
    float height = y2 - y1;
    bboxes.emplace_back(
        static_cast<int>(x1),
        static_cast<int>(y1),
        static_cast<int>(width),
        static_cast<int>(height)
    );
  }

  // Apply OpenCV NMS
  std::vector<int> keepIndices;
  cv::dnn::NMSBoxes(
      bboxes, scores,
      confidence_thresh, // score_threshold
      iou_thresh,        // nms_threshold
      keepIndices,
      1.f,               // eta
      keep_top_k         // top_k
  );

  // If no indices kept, nothing to draw
  if (keepIndices.empty()) {
    // Optionally, you could still overlay FPS text
    // but no faces will be drawn
    // Show FPS label in bottom-left:
    {
      const std::string label_fps = cv::format("Fps: %.2f", fps);
      int baseline = 0;
      cv::Size textSize = cv::getTextSize(label_fps,
                                          cv::FONT_HERSHEY_TRIPLEX,
                                          0.4, 1, &baseline);
      cv::rectangle(frame,
                    cv::Point(0, frame.rows - textSize.height - 6),
                    cv::Point(textSize.width + 2, frame.rows),
                    cv::Scalar(255, 255, 255),
                    cv::FILLED);

      cv::putText(frame, label_fps,
                  cv::Point(2, frame.rows - 4),
                  cv::FONT_HERSHEY_TRIPLEX,
                  0.4,
                  cv::Scalar(0, 0, 0),
                  1);
    }
    return;
  }

  // --------------------------------------------------
  // 3) Keep only NMS-filtered detections in 'faces'
  // --------------------------------------------------
  cv::Mat finalDetections(static_cast<int>(keepIndices.size()), faces.cols, CV_32F);
  for (size_t i = 0; i < keepIndices.size(); i++) {
    faces.row(keepIndices[i]).copyTo(finalDetections.row(static_cast<int>(i)));
  }

  // --------------------------------------------------
  // 4) Draw each face bounding box + 5 landmarks
  // --------------------------------------------------
  for (int r = 0; r < finalDetections.rows; r++)
  {
    float x1 = finalDetections.at<float>(r, 0);
    float y1 = finalDetections.at<float>(r, 1);
    float w = finalDetections.at<float>(r, 2);
    float h = finalDetections.at<float>(r, 3);

    // Landmarks: columns 4..13 => 5 pairs
    // finalDetections(r,4 + 2*k), finalDetections(r,5 + 2*k)
    float score = finalDetections.at<float>(r, 14);

    // Draw bounding box
    cv::rectangle(frame,
                  cv::Point2f(x1, y1),
                  cv::Point2f(x1+w, y1+h),
                  cv::Scalar(0, 255, 0),
                  2);

    // Draw 5 landmarks
    for (int k = 0; k < 5; k++) {
      float lmk_x = finalDetections.at<float>(r, 4 + 2*k);
      float lmk_y = finalDetections.at<float>(r, 4 + 2*k + 1);

      cv::circle(frame,
                 cv::Point2f(lmk_x, lmk_y),
                 2,
                 cv::Scalar(0, 0, 255),
                 cv::FILLED);
    }

    // Optionally display the score at top-left corner
    std::string txt = cv::format("%.2f", score);
    cv::putText(frame, txt, cv::Point((int)x1, (int)(y1 - 5)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255,255,255), 1);

  }

  // Display number of faces found:
  std::string num_face_txt = cv::format("number of faces: %i", finalDetections.rows);
  cv::putText(frame,num_face_txt,
              cv::Point(10,20),
              cv::FONT_HERSHEY_TRIPLEX,
              0.4,
              cv::Scalar(0,0,0),
              1);

  // --------------------------------------------------
  // 5) Show FPS label in bottom-left corner
  // --------------------------------------------------
  {
    const std::string label_fps = cv::format("Fps: %.2f", fps);
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(label_fps,
                                        cv::FONT_HERSHEY_TRIPLEX,
                                        0.4, 1, &baseline);
    cv::rectangle(frame,
                  cv::Point(0, frame.rows - textSize.height - 6),
                  cv::Point(textSize.width + 2, frame.rows),
                  cv::Scalar(255, 255, 255),
                  cv::FILLED);

    cv::putText(frame, label_fps,
                cv::Point(2, frame.rows - 4),
                cv::FONT_HERSHEY_TRIPLEX,
                0.4,
                cv::Scalar(0, 0, 0),
                1);
  }

}

} // re