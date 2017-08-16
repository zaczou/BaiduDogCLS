/*
 * center_distance_error_layer.cpp
 *
 *  Created on: Jun 22, 2017
 *      Author: lab-xiong.jiangfeng
 */

#include <algorithm>
#include <vector>

#include "caffe/layers/center_distance_error_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CenterDistanceErrorLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  total_stride_= this->layer_param_.center_distance_param().total_stride();
  total_shift_ = this->layer_param_.center_distance_param().total_shift();
  input_size_ = this->layer_param_.center_distance_param().input_size();
  this->param_propagate_down_.resize(this->blobs_.size(), false);
}
template <typename Dtype>
void CenterDistanceErrorLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	  vector<int> top_shape(0);
	  top[0]->Reshape(top_shape);
}
template <typename Dtype>
void CenterDistanceErrorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	  const Dtype* confMap_data = bottom[0]->cpu_data();//confMap
	  const Dtype* bbox = bottom[1]->cpu_data();//bbox
	  Dtype* center_error = top[0]->mutable_cpu_data();

	  int num = bottom[0]->num();
	  int width = bottom[0]->width();
	  int count = bottom[0]->count();
	  int dim = count / num;

	  for (int i = 0; i < num; ++i) {
		  //get the index of target location for each example
		  std::vector<std::pair<float,int> > pairs;
		  for(int k=0;k<dim;++k){
			  pairs.push_back(std::make_pair(confMap_data[i*dim +k],k));
		  }
		  //find argmax{score}
		  std::sort(pairs.begin(),pairs.end(),std::greater<std::pair<Dtype,int> >());

		  float y = pairs[0].second/width;
		  float x = pairs[0].second%width;

		  float gt_x = (bbox[i*4]+bbox[i*4+2])/2.0;
		  float gt_y = (bbox[i*4+1]+bbox[i*4+3])/2.0;

		  y = (y* total_stride_ + total_shift_)/input_size_;//(0-1)
		  x = (x*total_stride_ + total_shift_)/input_size_;//(0-1)

		  center_error[0] = center_error[0] + std::sqrt(std::pow((gt_x-x),2)+std::pow((gt_y-y),2));
	  }
	  center_error[0]=center_error[0]/Dtype(num);//average over batch
}
INSTANTIATE_CLASS(CenterDistanceErrorLayer);
REGISTER_LAYER_CLASS(CenterDistanceError);
}  // namespace caffe
