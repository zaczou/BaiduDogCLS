#include <vector>

#include "caffe/layers/triplet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
										  const vector<Blob<Dtype>*>& top){
	 int count = bottom[0]->count();

  //bottom[0] achor
  //bottom[1]: pos
  //bottom[2]: neg

  caffe_gpu_sub(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
            diff_same_class_.mutable_gpu_data());
  caffe_gpu_sub(count, bottom[0]->gpu_data(), bottom[2]->gpu_data(),
            diff_diff_class_.mutable_gpu_data());


  
  Dtype loss = 0;
  Dtype diff_same_class_l2,diff_diff_class_l2;
  for (int v = 0; v < batch_size_; ++v) {
  	caffe_gpu_dot(vec_dimension_,diff_same_class_.gpu_data() + v * vec_dimension_,diff_same_class_.gpu_data() + v * vec_dimension_,&diff_same_class_l2);
  	caffe_gpu_dot(vec_dimension_,diff_diff_class_.gpu_data() + v * vec_dimension_,diff_diff_class_.gpu_data() + v * vec_dimension_,&diff_diff_class_l2);
  	
    vec_loss_[v] = alpha_ + diff_same_class_l2 - diff_diff_class_l2;
    vec_loss_[v] = std::max(Dtype(0), vec_loss_[v]);
    loss += vec_loss_[v];
  }
  
  
  loss /= batch_size_ * Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;  
//   std::cout<<"Debug info loss= "<<loss<<std::endl;
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>&top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	
  const Dtype scale = top[0]->cpu_diff()[0] / bottom[0]->num();
  const int n = bottom[0]->count();

  caffe_gpu_sub(n, diff_same_class_.gpu_data(), diff_diff_class_.gpu_data(),
            bottom[0]->mutable_gpu_diff());
  caffe_gpu_scal(n, scale, bottom[0]->mutable_gpu_diff());

  caffe_gpu_scale(n, -scale, diff_same_class_.gpu_data(),
                  bottom[1]->mutable_gpu_diff());

  caffe_gpu_scale(n, scale, diff_diff_class_.gpu_data(),
                  bottom[2]->mutable_gpu_diff());

  for (int v = 0; v < batch_size_; ++v) {
    if (vec_loss_[v] == 0) {
      caffe_gpu_set(vec_dimension_, Dtype(0),
                bottom[0]->mutable_gpu_diff() + v * vec_dimension_);
      caffe_gpu_set(vec_dimension_, Dtype(0),
                bottom[1]->mutable_gpu_diff() + v * vec_dimension_);
      caffe_gpu_set(vec_dimension_, Dtype(0),
                bottom[2]->mutable_gpu_diff() + v * vec_dimension_);
    }
  }
	
	
}
INSTANTIATE_LAYER_GPU_FUNCS(TripletLossLayer);
}// nameapace caffe