//
// Created by pkyle on 12/26/24.
//

#include "pose_estimation.h"
#include "opencv2/highgui.hpp"

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
  // Create Color Camera Node:
  cam_rgb_ = pipeline_.create<dai::node::ColorCamera>();
  cam_rgb_->setPreviewSize(kRgbWidth,kRgbHeight);
  cam_rgb_->setBoardSocket(dai::CameraBoardSocket::CAM_A);
  cam_rgb_->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
  cam_rgb_->setInterleaved(false);

  // Create Color Camera Output:
  xout_rgb_ = pipeline_.create<dai::node::XLinkOut>();
  xout_rgb_->setStreamName("rgb");
  cam_rgb_->preview.link(xout_rgb_->input);


  // Create Image Manip Node:
  manip_ = pipeline_.create<dai::node::ImageManip>();
  manip_->initialConfig.setResize(kNnWidth,kNnHeight);
  manip_->initialConfig.setFrameType(dai::RawImgFrame::Type::BGR888p);
  manip_->setKeepAspectRatio(false);

  // Link Camera Output to ImageManip input
  cam_rgb_->preview.link(manip_->inputImage);

  // Create Neural Network Node
  nn_ = pipeline_.create<dai::node::NeuralNetwork>();
  dai::OpenVINO::Blob blob("/home/pkyle/reflective_encounters/face_pose/pose_estimation/face_detection_yunet_160x120.blob");
  nn_->setBlob(blob);
  nn_->setNumInferenceThreads(2);
  nn_->input.setBlocking(false);

  // Link manip outut to nn input
  manip_->out.link(nn_->input);

  // Create XLinkOut to get NN detections on the host
  xout_nn_ = pipeline_.create<dai::node::XLinkOut>();
  xout_nn_->setStreamName("nn");
  nn_->out.link(xout_nn_->input);

  // Connect to device
  device_ = std::make_shared<dai::Device>(pipeline_, dai::UsbSpeed::SUPER);
  LOG(INFO) << "Usb Speed: " << device_->getUsbSpeed();

  // Bootloader Version
  if(device_->getBootloaderVersion()){
    LOG(INFO) << "Bootloader Version: " << device_->getBootloaderVersion()->toString();
  }

  // Device Name
  LOG(INFO) << "Device Name: " << device_->getDeviceName() << " Product Name: " << device_->getProductName();

  // Output Queue, used to get rgb frames
  q_rgb_ = device_->getOutputQueue("rgb", 4, false);
  q_det_ = device_->getOutputQueue("nn", 4, false);

}
void PoseEstimation::DisplayVideo() {
  cv::namedWindow("rgb");
  while( ! stop_.load()){
    // Wait for Camera Frame
    auto in_rgb = q_rgb_->get<dai::ImgFrame>();
    auto frame = in_rgb->getCvFrame();

    // Try to get NN output
    auto in_det = q_det_->tryGet<dai::NNData>();

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
      // Get Confidence
      std::vector<float> face_conf;
      face_conf.reserve(1076);
      for(int i = 0; i < 1076; i++){
        auto cls_score = conf_mat.at<float>(i,1);
        auto iou_score = iou_mat.at<float>(i,0);
        auto face_confidence = std::min(cls_score, iou_score);
        if(face_confidence > 0.5){
          face_conf.push_back(face_confidence);
        }
      }
      if(!face_conf.empty()){
        std::cout << "\rmax face conf: " << *std::max_element(face_conf.begin(), face_conf.end())<< std::flush;
      }

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
  catch(...){
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

PoseEstimation::PriorBox::PriorBox() : input_shape(cv::Size(kNnWidth,kNnHeight)),
                                       output_shape(cv::Size(kRgbWidth, kRgbHeight)),
                                       strides({8,16,32}),
                                       min_sizes({
                                         {10.f, 16.f, 24.f},
                                         {32.f, 48.f, 64.f},
                                         {96.f, 128.f, 160.f}
                                       })
{
  BuildPriors();
}
cv::Mat PoseEstimation::PriorBox::Decode(const cv::Mat &loc,
                                         const cv::Mat &conf,
                                         const cv::Mat &iou,
                                         float conf_threshold,
                                         int topK) {
  int N = loc.rows;
  if(N == 0){
    return cv::Mat();
  }

  // 1. Extract second column 'conf' and  first column of iou
  std::vector<float> cls_scores(N), iou_scores(N);
  for(int i = 0; i<N; i++){
    cls_scores[i] = conf.at<float>(i,1);
    iou_scores[i] = iou.at<float>(i,0);
  }

  // 2. Compute face_conf = min between the two scores
  std::vector<float>face_conf(N);
  for(int i = 0; i < N; i++){
    face_conf[i] = std::min(cls_scores[i], iou_scores[i]);
  }

  // 3. For each prior, decode bounding box;
  std::vector<float> cx_vec(N), cy_vec(N), w_vec(N), h_vec(N);
  for(int i = 0; i < N; i++){
    float dx = loc.at<float>(i,0);
    float dy = loc.at<float>(i,1);
    float dw = loc.at<float>(i,2);
    float dh = loc.at<float>(i,3);

    float cx0 = priors[i][0];
    float cy0 = priors[i][1];
    float sKx = priors[i][2];
    float sKy = priors[i][3];

    float cx = dx* 0.1f* sKx + cx0;
    float cy = dy* 0.1f* sKy + cy0;
    float w = std::exp(dw*0.2f)*sKx;
    float h = std::exp(dh*0.2f)*sKy;

    cx_vec[i] = cx;
    cy_vec[i] = cy;
    w_vec[i] = w;
    h_vec[i] = h;
  }

  // 4. Decode landmarks
  std::vector<std::array<float,10>> landmarks_vec(N);
  for(int i = 0; i < N; i++){
    float cx0 = priors[i][0];
    float cy0 = priors[i][1];
    float sKx = priors[i][2];
    float sKy = priors[i][3];
    // for loc row i, columns [4 - 13]
    for (int j = 0; j < 5; j++){
      float lmk_x = loc.at<float>(i, 4+2*j);
      float lmk_y = loc.at<float>(i, 5+2*j);
      // scale
      lmk_x = lmk_x*0.1f*sKx + cx0;
      lmk_y = lmk_y*0.1f*sKy + cy0;
      landmarks_vec[i][2*j] = lmk_x;
      landmarks_vec[i][2*j+1] = lmk_y;
    }
  }

  // 5. Filter by Confidence Threshold
  std::vector<int> valid_idx;
  valid_idx.reserve(N);
  for(int i = 0; i < N; i++){
    if(face_conf[i] > conf_threshold){
      valid_idx.push_back(i);
    }
  }
  if(valid_idx.empty()){
    return cv::Mat();
  }

  // 6. If more than max number of faces (top k), lower confidence guesses)
  if((int)valid_idx.size() > topK){
    std::sort(valid_idx.begin(), valid_idx.end(), [&](int a, int b){
      return face_conf[a] > face_conf[b];
    });
    valid_idx.resize(topK);
  }

  // 7 Scale to output image size
  auto out_w = (float)output_shape.width;
  auto out_h = (float)output_shape.height;
  cv::Mat decoded(valid_idx.size(), 13, CV_32F);
  for(int r = 0; r < (int)valid_idx.size(); r++){
    int i = valid_idx[r];
    float cx = cx_vec[i]* out_w;
    float cy = cy_vec[i]* out_h;
    cx = std::clamp(cx, 0.f, out_w-1);
    cy = std::clamp(cy, 0.f, out_h-1);

    decoded.at<float>(r, 0) = cx;
    decoded.at<float>(r, 1) = cy;

    // landmarks
    for (int j = 0; j< 5; j++){
      float lx = landmarks_vec[i][2*j]*out_w;
      float ly = landmarks_vec[i][2*j+1]*out_h;
      decoded.at<float>(r,4 + 2*j) = lx;
      decoded.at<float>(r,4 + 2*j+1) = ly;
    }

    // confidence
    decoded.at<float>(r,13) = face_conf[i];
  }
  return decoded;
}
void PoseEstimation::PriorBox::BuildPriors() {
  for(size_t s = 0; s < strides.size(); s++){
    int stride = strides[s];
    const auto& min_sizes_set = min_sizes[s];
    int out_w = (int)std::ceil((float)input_shape.width / (float)stride);
    int out_h = (int)std::ceil((float)input_shape.height / (float)stride);
    for(int i = 0; i < out_h; i ++){
      for(int j = 0; j < out_w; j++){
        for(auto min_size : min_sizes_set){
          float s_kx = min_size / (float)input_shape.width;
          float s_ky = min_size / (float)input_shape.height;
          float cx = ((float)j+0.5f)* (float)stride / (float)input_shape.width;
          float cy = ((float)i+0.5f)* (float)stride / (float)input_shape.height;
          priors.emplace_back(cx,cy, s_kx,s_ky);
        }
      }
    }
  }

}

} // re