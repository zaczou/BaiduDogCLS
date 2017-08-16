/*
 * structured_output_loss_layer.hpp
 *
 *  Created on: Jun 9, 2017
 *      Author: lab-xiong.jiangfeng
 */

#ifndef CAFFE_STRUCTURED_OUTPUT_LOSS_LAYER_HPP_
#define CAFFE_STRUCTURED_OUTPUT_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
template <typename Dtype>
class StructuredOutputLossLayer : public LossLayer<Dtype> {
 public:
  explicit StructuredOutputLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "StructuredOutputLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<int> max_conf_index_;
  Blob<Dtype> signed_slack_variable_;

};
}  // namespace caffe
#endif /* CAFFE_STRUCTURED_OUTPUT_LOSS_LAYER_HPP_ */
