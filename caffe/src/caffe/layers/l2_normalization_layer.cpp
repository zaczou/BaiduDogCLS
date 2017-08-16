#include <vector>
#include <algorithm>

#include "caffe/layers/l2_normalization_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/device_alternate.hpp"

namespace caffe{
	
template <typename Dtype>
void L2NormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
	vector<int> top_shape = bottom[0]->shape();
	top[0]->Reshape(top_shape);
		
	vector<int> norm_inv_shape(1);
	norm_inv_shape[0] = bottom[0]->shape()[0];
	norm_inv_.Reshape(norm_inv_shape);
	}

template <typename Dtype>
void L2NormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	Dtype* norm_inv_data = norm_inv_.mutable_cpu_data();
	int dim = bottom[0]->count(1);
	int num = bottom[0]->shape()[0];
		
	caffe_powx(num*dim,bottom_data,Dtype(2),top_data);
	for(int n=0;n < num;n++){
		norm_inv_data[n] = pow(caffe_cpu_asum<Dtype>(dim, top_data+n*dim),Dtype(-0.5));
		caffe_cpu_scale(dim, norm_inv_data[n], bottom_data + n*dim ,top_data + n*dim);
	}
}

template <typename Dtype>
void L2NormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom){
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	
	const Dtype* top_data = top[0]->cpu_data();
	const Dtype* norm_inv_data = norm_inv_.cpu_data();
	int dim = bottom[0]->count(1);
	int num = bottom[0]->shape()[0];
	
	Dtype diff_scale;
	for(int n=0;n<num;n++){
		diff_scale = norm_inv_data[n] * (1.0 - caffe_cpu_dot(dim,top_data+n*dim,top_data+n*dim));
		caffe_cpu_scale(dim , diff_scale,top_diff+n*dim,bottom_diff);
	}
}

#ifdef CPU_ONLY
STUB_GPU(L2NormLayer);
#endif
	
INSTANTIATE_CLASS(L2NormLayer);
REGISTER_LAYER_CLASS(L2Norm);
	
}//namespace caffe
