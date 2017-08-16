#include <vector>

#include "caffe/layers/boost_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BoostDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
        top[1]->mutable_gpu_data());
  }
  if (this->output_weights_) {
    // Reshape to loaded weights.
    top[2]->ReshapeLike(batch->weight_);
    // Copy the weights.
    caffe_copy(batch->weight_.count(), batch->weight_.gpu_data(),
        top[2]->mutable_gpu_data());
  }
  if (this->output_ids_) {
    // Reshape to loaded ids.
    top[3]->ReshapeLike(batch->id_);
    // Copy the ids.
    caffe_copy(batch->id_.count(), batch->id_.gpu_data(),
        top[3]->mutable_gpu_data());
  }

  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BoostDataLayer);

}  // namespace caffe
