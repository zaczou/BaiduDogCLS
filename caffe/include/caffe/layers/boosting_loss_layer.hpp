#ifndef CAFFE_BOOSTING_LOSS_LAYER_HPP_
#define CAFFE_BOOSTING_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class BoostingLossLayer : public LossLayer<Dtype> {
 public:
  LayerParameter layer_param_;
  std::string boosting_type_;
  explicit BoostingLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), layer_param_(param) {
        if (param.has_boosting_param()) {
          boosting_type_ = param.boosting_param().type();
        }
      }
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);


  virtual inline const char* type() const { return "BoostingLoss"; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }
  virtual inline int ExactNumBottomBlobs() const { return 4; }


 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  vector<Dtype> lambda_;
};

}  // namespace caffe

#endif  // CAFFE_BOOSTING_LAYER_HPP_
