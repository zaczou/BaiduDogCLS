#include <vector>
#include <algorithm>

#include "caffe/layers/l2_normalization_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/device_alternate.hpp"

namespace caffe{
template <typename Dtype>
void L2NormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	Dtype* norm_inv_data = norm_inv_.mutable_gpu_data();
	int dim = bottom[0]->count(1);
	int num = bottom[0]->shape()[0];
		
	caffe_gpu_powx(num*dim,bottom_data,Dtype(2),top_data);
	for(int n=0;n < num;n++){
		caffe_gpu_asum<Dtype>(dim, top_data+n*dim,norm_inv_data+n);
		norm_inv_data[n]= pow(norm_inv_data[n],Dtype(-0.5));
		caffe_gpu_scale(dim, norm_inv_data[n], bottom_data + n*dim ,top_data + n*dim);
	}
}

template <typename Dtype>
void L2NormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom){
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	
	const Dtype* top_data = top[0]->gpu_data();
	const Dtype* norm_inv_data = norm_inv_.gpu_data();
	int dim = bottom[0]->count(1);
	int num = bottom[0]->shape()[0];
	
	Dtype diff_scale,top_ip;
	for(int n=0;n<num;n++){
		caffe_gpu_dot(dim,top_data+n*dim,top_data+n*dim,&top_ip);
		diff_scale = norm_inv_data[n] * (1.0 -top_ip);
		caffe_gpu_scale(dim , diff_scale,top_diff+n*dim,bottom_diff);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(L2NormLayer);
}//namespace caffe