//
// Created by pkyle on 1/19/25.
//

#ifndef FACE_POSE_PROJECT_POSE_ESTIMATION_PRIOR_BOX_H_
#define FACE_POSE_PROJECT_POSE_ESTIMATION_PRIOR_BOX_H_
#pragma once

#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdlib>   // For std::exit()
#include <stdexcept>
#include <iostream>
#include <vector>
#include <array>

/**
 * @brief Direct C++ translation of ShiqiYu/libfacedetection "priorbox.py".
 *
 * The original Python code:
 *   https://github.com/ShiqiYu/libfacedetection/blob/master/opencv_dnn/python/priorbox.py
 */
class PriorBox {
 public:
  /**
   * @param inputShape  (width, height) for the NN input. Default (320, 240).
   * @param outputShape (width, height) for the final frame where boxes/landmarks are drawn.
   *                    Default (320, 240).
   * @param variance    Defaults to [0.1, 0.2].
   */
  PriorBox(
      const cv::Size2i& inputShape  = cv::Size2i(320, 240),
      const cv::Size2i& outputShape = cv::Size2i(320, 240),
      const std::vector<float>& variance = {0.1f, 0.2f}
  )
  {
    // ------------------------------------------------
    // 1) Match the Python defaults
    // ------------------------------------------------
    minSizes_ = {
        { 10.f, 16.f, 24.f },
        { 32.f, 48.f      },
        { 64.f, 96.f      },
        { 128.f, 192.f, 256.f }
    };
    steps_ = { 8, 16, 32, 64 };

    in_w_  = inputShape.width;   // self.in_w
    in_h_  = inputShape.height;  // self.in_h
    out_w_ = outputShape.width;  // self.out_w
    out_h_ = outputShape.height; // self.out_h

    variance_ = variance;  // self.variance = [0.1, 0.2]

    // Python check: for ii in range(4), steps[ii] == 2^(3+ii)
    for (int ii = 0; ii < 4; ++ii) {
      if (steps_[ii] != (1 << (ii + 3))) {  // 2^(3+ii)
        std::cerr << "steps must be [8,16,32,64]\n";
        std::exit(1);
      }
    }

    // ------------------------------------------------
    // 2) Compute feature_map_{2th,3th,4th,5th,6th}
    //    EXACT integer divisions as in Python
    // ------------------------------------------------
    {
      // self.feature_map_2th = [(in_h+1)//2 //2, (in_w+1)//2 //2]
      int fm2_h = (in_h_ + 1) / 2;  // integer division
      fm2_h = fm2_h / 2;
      int fm2_w = (in_w_ + 1) / 2;
      fm2_w = fm2_w / 2;
      feature_map_2th_ = { fm2_h, fm2_w };

      // self.feature_map_3th = [feature_map_2th[0]//2, feature_map_2th[1]//2]
      feature_map_3th_ = {
          feature_map_2th_[0] / 2,
          feature_map_2th_[1] / 2
      };
      // self.feature_map_4th = ...
      feature_map_4th_ = {
          feature_map_3th_[0] / 2,
          feature_map_3th_[1] / 2
      };
      // self.feature_map_5th = ...
      feature_map_5th_ = {
          feature_map_4th_[0] / 2,
          feature_map_4th_[1] / 2
      };
      // self.feature_map_6th = ...
      feature_map_6th_ = {
          feature_map_5th_[0] / 2,
          feature_map_5th_[1] / 2
      };
    }

    // self.feature_maps = [feature_map_3th, feature_map_4th, feature_map_5th, feature_map_6th]
    featureMaps_.push_back(feature_map_3th_);
    featureMaps_.push_back(feature_map_4th_);
    featureMaps_.push_back(feature_map_5th_);
    featureMaps_.push_back(feature_map_6th_);

    // ------------------------------------------------
    // 3) Generate priors => self.priors
    // ------------------------------------------------
    generatePriors();
  }

  /**
   * @brief Generate the anchors (priors_) exactly as Python's generate_priors().
   *
   * Anchors are stored as [cx, cy, w, h] in normalized coords.
   */
  void generatePriors() {
    // anchors = np.empty(shape=[0,4])
    // for k,f in enumerate(self.feature_maps):
    //     min_sizes = self.min_sizes[k]
    //     for i,j in product(range(f[0]), range(f[1])):
    //         for min_size in min_sizes:
    //             s_kx = min_size / in_w_
    //             s_ky = min_size / in_h_
    //             cx = (j+0.5)*steps_[k]/in_w_
    //             cy = (i+0.5)*steps_[k]/in_h_
    //             anchors = vstack( anchors, [cx,cy,s_kx,s_ky] )
    // return anchors

    for (size_t k = 0; k < featureMaps_.size(); ++k) {
      auto& f = featureMaps_[k];     // [f_h, f_w]
      auto& min_sizes = minSizes_[k];
      int step = steps_[k];

      int f_h = f[0];  // corresponds to range(f[0]) in Python => i
      int f_w = f[1];  // corresponds to range(f[1]) in Python => j

      for (int i = 0; i < f_h; i++) {
        for (int j = 0; j < f_w; j++) {
          for (float min_size : min_sizes) {
            float s_kx = min_size / static_cast<float>(in_w_);
            float s_ky = min_size / static_cast<float>(in_h_);
            float cx   = ((float)j + 0.5f) * step / (float)in_w_;
            float cy   = ((float)i + 0.5f) * step / (float)in_h_;

            priors_.push_back({cx, cy, s_kx, s_ky});
          }
        }
      }
    }
  }

  /**
   * @brief Decode bounding boxes + landmarks, filtering out low scores.
   *
   * Mirrors the Python method:
   *   decode(self, loc, conf, iou, ignore_score=0.6)
   *
   * The shapes must be:
   *   loc:  (N, 14) => [x_c, y_c, w, h, (5 pairs of landmarks)]
   *   conf: (N,  2) => [bg_conf, face_conf]
   *   iou:  (N,  1) => [iou]
   *
   * The final output is a (M x 15) Mat:
   *   [x1, y1, w, h, lm1_x, lm1_y, ..., lm5_x, lm5_y, score]
   *   where M <= N (only those passing ignore_score).
   *
   * Notes:
   *  - The bounding box is returned as (x1, y1, w, h) in **pixel** coords.
   *  - Landmarks are also scaled to pixel coords in the output frame.
   */
  cv::Mat decode(const cv::Mat& loc,
                 const cv::Mat& conf,
                 const cv::Mat& iou,
                 float ignore_score = 0.6f) const
  {
    // -------------------------------
    // 1) Validate input shapes
    // -------------------------------
    if (loc.empty() || conf.empty() || iou.empty()) {
      return cv::Mat();
    }
    int num_priors = static_cast<int>(priors_.size());
    if (loc.rows != num_priors ||
        conf.rows != num_priors ||
        iou.rows  != num_priors)
    {
      throw std::runtime_error("loc/conf/iou shape mismatch with number of priors.");
    }
    if (loc.cols != 14 || conf.cols != 2 || iou.cols != 1) {
      throw std::runtime_error("Expected loc=(N,14), conf=(N,2), iou=(N,1).");
    }

    // -------------------------------
    // 2) Compute final face scores
    //    = sqrt(conf_face * iou_face),
    //    clamp iou to [0,1], then filter
    // -------------------------------
    std::vector<float> scores(num_priors, 0.f);
    for (int r = 0; r < num_priors; r++) {
      float cls_score = conf.at<float>(r, 1); // face conf
      float iou_score = iou.at<float>(r, 0);
      if (iou_score < 0.f) iou_score = 0.f;
      if (iou_score > 1.f) iou_score = 1.f;

      float final_score = std::sqrt(cls_score * iou_score);
      scores[r] = final_score;
    }

    // Find indices where score > ignore_score
    std::vector<int> idxs;
    idxs.reserve(num_priors);
    for (int r = 0; r < num_priors; r++) {
      if (scores[r] > ignore_score) {
        idxs.push_back(r);
      }
    }

    // If none pass, return empty
    if (idxs.empty()) {
      return cv::Mat();
    }

    // -------------------------------
    // 3) For each retained index:
    //      - decode bounding box
    //      - decode 5 landmarks
    //      - store in output vector
    // -------------------------------
    std::vector<std::array<float, 15>> outVec;
    outVec.reserve(idxs.size());

    float var0 = variance_[0]; // 0.1
    float var1 = variance_[1]; // 0.2

    for (int r : idxs) {
      // prior = [cx, cy, w, h] in normalized coords
      float cx_p = priors_[r][0];
      float cy_p = priors_[r][1];
      float w_p  = priors_[r][2];
      float h_p  = priors_[r][3];

      // loc(r,:) => [dx, dy, dw, dh, lmk(10)]
      float dx = loc.at<float>(r, 0);
      float dy = loc.at<float>(r, 1);
      float dw = loc.at<float>(r, 2);
      float dh = loc.at<float>(r, 3);

      // bboxes = np.hstack((
      //   priors[:,0:2] + loc[:,0:2]*var0*priors[:,2:4],
      //   priors[:,2:4]*exp(loc[:,2:4]*var1)
      // ))
      float x_c = cx_p + dx * var0 * w_p;         // x center
      float y_c = cy_p + dy * var0 * h_p;         // y center
      float w   = w_p * std::exp(dw * var1);      // width  (normalized)
      float h   = h_p * std::exp(dh * var1);      // height (normalized)

      // (x_c, y_c, w, h) -> (x1, y1, w, h)
      // x1 = x_c - w/2
      // y1 = y_c - h/2
      float x1 = x_c - 0.5f * w;
      float y1 = y_c - 0.5f * h;

      // scale to output: multiply [x1, y1, w, h] by [out_w, out_h, out_w, out_h]
      x1 *= out_w_;
      y1 *= out_h_;
      w  *= out_w_;
      h  *= out_h_;

      // landmarks => for each of the 5 points
      //   lmk_x = cx_p + loc_x * var0 * w_p
      //   lmk_y = cy_p + loc_y * var0 * h_p
      // then scale => out_w_, out_h_
      std::array<float, 10> lmkPts{};
      for (int k = 0; k < 5; k++) {
        float lx = loc.at<float>(r, 4 + 2 * k + 0);
        float ly = loc.at<float>(r, 4 + 2 * k + 1);

        float lmk_x = cx_p + lx * var0 * w_p;
        float lmk_y = cy_p + ly * var0 * h_p;

        lmk_x *= out_w_;
        lmk_y *= out_h_;

        lmkPts[2*k + 0] = lmk_x;
        lmkPts[2*k + 1] = lmk_y;
      }

      float sc = scores[r];

      // final row => [x1, y1, w, h, lmk(10), score]
      std::array<float, 15> row;
      row[0] = x1;   row[1] = y1;
      row[2] = w;    row[3] = h;
      for (int m = 0; m < 10; m++) {
        row[4 + m] = lmkPts[m];
      }
      row[14] = sc;

      outVec.push_back(row);
    }

    // -------------------------------
    // 4) Convert to cv::Mat (M x 15)
    // -------------------------------
    cv::Mat out((int)outVec.size(), 15, CV_32F);
    for (int i = 0; i < (int)outVec.size(); i++) {
      for (int c = 0; c < 15; c++) {
        out.at<float>(i, c) = outVec[i][c];
      }
    }
    return out;
  }

  /**
   * @return All generated priors [cx, cy, w, h] (normalized).
   */
  const std::vector<std::array<float, 4>>& getPriors() const {
    return priors_;
  }

 private:
  // Python code data
  // self.min_sizes = [[10,16,24], [32,48], [64,96], [128,192,256]]
  std::vector<std::vector<float>> minSizes_;
  // self.steps = [8,16,32,64]
  std::vector<int> steps_;

  // from input_shape=(in_w, in_h), output_shape=(out_w, out_h)
  int in_w_,  in_h_;
  int out_w_, out_h_;

  // self.variance = [0.1, 0.2]
  std::vector<float> variance_;

  // feature_map_2th,3th,4th,5th,6th
  std::array<int,2> feature_map_2th_;
  std::array<int,2> feature_map_3th_;
  std::array<int,2> feature_map_4th_;
  std::array<int,2> feature_map_5th_;
  std::array<int,2> feature_map_6th_;

  // self.feature_maps = [feature_map_3th,4th,5th,6th]
  std::vector<std::array<int,2>> featureMaps_;

  // self.priors => [cx, cy, w, h] in normalized coords
  std::vector<std::array<float,4>> priors_;
};


#endif //FACE_POSE_PROJECT_POSE_ESTIMATION_PRIOR_BOX_H_
