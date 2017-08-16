#ifndef CAFFE_L2NORM_LAYER_
#define CAFFE_L2NORM_LAYER_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
	
template <typename Dtype>
class L2NormLayer: public Layer<Dtype>{
 public:
	explicit L2NormLayer(const LayerParameter& param)
		:Layer<Dtype>(param){}
	
	virtual inline const char* type() const{return "L2Norm";}
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual inline int ExtractNumBottomBlobs() const {return 1;}
	virtual inline int ExtractNumTopBlobs() const{return 1;}
	
 protected:
	
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom);
	
	Blob<Dtype> norm_inv_;
};
	
}// namespace caffe


#endif