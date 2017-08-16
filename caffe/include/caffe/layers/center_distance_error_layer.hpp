/*
 * center_distance_error_layer.hpp
 *
 *  Created on: Jun 22, 2017
 *      Author: lab-xiong.jiangfeng
 */

#ifndef CAFFE_CENTER_DISTANCE_ERROR_LAYER_HPP_
#define CAFFE_CENTER_DISTANCE_ERROR_LAYER_HPP_
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
template <typename Dtype>
class CenterDistanceErrorLayer : public Layer<Dtype> {
 public:
  explicit CenterDistanceErrorLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual inline const char* type() const { return "CenterDistanceError"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

  float total_stride_;
  float total_shift_;
  float input_size_;
};
}  // namespace caffe

#endif /* CAFFE_CENTER_DISTANCE_ERROR_LAYER_HPP_ */
