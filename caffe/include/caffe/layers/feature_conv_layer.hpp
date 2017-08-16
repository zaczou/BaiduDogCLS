#ifndef CAFFE_FEATURE_CONV_LAYER_HPP_
#define CAFFE_FEATURE_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

/**
 * @brief Convolves two blobs,
 *        and (optionally) adds biases.
 */
template <typename Dtype>
class FeatureConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:

  explicit FeatureConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
  
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "FeatureConvolution"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline bool EqualNumBottomTopBlobs() const {return false;}

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();
};

}  // namespace caffe

#endif  // CAFFE_FEATURE_CONV_LAYER_HPP_