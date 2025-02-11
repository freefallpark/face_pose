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


PoseEstimation::PoseEstimation() : stop_(false), pipeline_(), device_(){}
PoseEstimation::~PoseEstimation() {
  Stop();
  if(pose_thread_.joinable()){
    pose_thread_.join();
  }
}
void PoseEstimation::Init(){
  // Create Neural Network Node
  nn_ = pipeline_.create<dai::node::NeuralNetwork>();
  dai::OpenVINO::Blob blob("face_detection_yunet_160x120.blob");
  nn_->setBlob(blob);
  nn_->setNumInferenceThreads(2);
  nn_->input.setBlocking(false);

  // Create Camera Node and set it up:
  cam_ = pipeline_.create<dai::node::ColorCamera>();
  cam_->setBoardSocket(dai::CameraBoardSocket::CAM_A);
  cam_->setPreviewSize(kRgbWidth, kRgbHeight);
  cam_->setInterleaved(false);
  cam_->setFps(60);
  cam_->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);

  // Create NN Image Manip Node:
  manip_nn_ = pipeline_.create<dai::node::ImageManip>();
  manip_nn_->initialConfig.setResize(kNnWidth, kNnHeight);
  manip_nn_->initialConfig.setFrameType(dai::RawImgFrame::Type::BGR888p);

  // Create outputs
  xout_ = pipeline_.create<dai::node::XLinkOut>();
  xout_->setStreamName("rgb");
  xout_nn_ = pipeline_.create<dai::node::XLinkOut>();
  xout_nn_->setStreamName("nn");

  // Link Camera Output to manip and xout
  cam_->preview.link(manip_nn_->inputImage);
  cam_->preview.link(xout_->input);

  // Link manip outut to nn input
  manip_nn_->out.link(nn_->input);
  nn_->out.link(xout_nn_->input);

  // Connect to device
  LOG(INFO) << "Attempting to Create Device on Pipeline";
  device_ = std::make_shared<dai::Device>(pipeline_);
  LOG(INFO) << "Usb Speed: " << device_->getUsbSpeed();

  // Bootloader Version
  if(device_->getBootloaderVersion()){
    LOG(INFO) << "Bootloader Version: " << device_->getBootloaderVersion()->toString();
  }

  // Device Name
  LOG(INFO) << "Device Name: " << device_->getDeviceName() << " Product Name: " << device_->getProductName();

  // Output Queue, used to get rgb frames
  q_cam_ = device_->getOutputQueue("rgb", 4, false);
  q_nn_ = device_->getOutputQueue("nn", 4, false);

}
void PoseEstimation::DisplayVideo() {
  cv::namedWindow("rgb");
  while( ! stop_.load()){
    // Wait for Camera Frame
    auto in_rgb = q_cam_->get<dai::ImgFrame>();
    auto frame = in_rgb->getCvFrame();

    // Try to get NN output
    auto in_det = q_nn_->tryGet<dai::NNData>();

    // Parse output
    std::vector<float> conf, iou, loc;
    cv::Mat conf_mat, iou_mat, loc_mat, detected_faces;
    if(in_det != nullptr){
      conf = in_det->getLayerFp16("conf");
      iou = in_det->getLayerFp16("iou");
      loc = in_det->getLayerFp16("loc");
      conf_mat = cv::Mat(1076,2, CV_32F, conf.data());
      iou_mat = cv::Mat(1076,1, CV_32F, iou.data());
      loc_mat = cv::Mat(1076,14, CV_32F, loc.data());

      // Decode
      auto pb = PriorBox(cv::Size2i(kNnWidth,kNnHeight),
                         cv::Size2i(kRgbWidth, kRgbHeight));
      auto faces = pb.decode(loc_mat, conf_mat, iou_mat, 0.5);

      DrawFaces(frame,faces);

    }

    cv::imshow("rgb", frame);
    cv::waitKey(1);
  }
  cv::destroyAllWindows();
}
bool PoseEstimation::Run() {
  // Initialize
  try{
    Init();
  }
  catch(std::runtime_error& e){
    LOG(WARNING) << "Exception Caught during Initialization: " << e.what();
    Stop();
    return false;
  }

  // Start Video Display thread
  pose_thread_ = std::thread(&PoseEstimation::DisplayVideo, this);

  LOG(INFO) << "Successfully started Video Display";

  return true;
}
void PoseEstimation::Stop() {
  stop_.store(true);
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