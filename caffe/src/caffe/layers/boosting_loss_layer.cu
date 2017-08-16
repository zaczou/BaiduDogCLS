#include <vector>
#include <cfloat>
#include <iostream>

#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/boosting_loss_layer.hpp"

#define SHOW(x)  LOG(INFO) << #x << " " << x

namespace caffe {

template <typename Dtype>
void BoostingLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top); return;

  const Dtype* act = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* weights = bottom[2]->cpu_data();

  int n = bottom[0]->shape()[0]; // batch_size
  int m = bottom[0]->shape()[1]; // num_classes

  int num = n;
  int dim = m;

  Dtype loss = 0;
  // sqr
  if (boosting_type_ == "sqr") {
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < m; ++j)
        loss += (act[i*m+j] + weights[i*m+j]) * (act[i*m+j] + weights[i*m+j]);
  } else if (boosting_type_ == "softmax") {
    for (int i = 0; i < n; ++i) {
      Dtype temp = 0.0;
      int label_index = i * m + static_cast<int>(label[i]);
      for ( int j = 0; j < m; ++j) {
        //temp += exp(std::min((Dtype)60., act[i * m + j] - act[label_index]));
        temp += exp(act[i * m + j] - act[label_index]);
      }
      loss += log(std::max(temp, Dtype(FLT_MIN)));
    }
  } else if (boosting_type_ == "ws") {
    for (int i = 0; i < n; ++i) {
      Dtype temp = 0.0;
      int label_index = i * m + static_cast<int>(label[i]);
      for ( int j = 0; j < m; ++j)
        temp += weights[i*m+j] * exp(act[i * m + j] - act[label_index]);
      loss += log(std::max(temp, Dtype(FLT_MIN)));
    }
  } else if (boosting_type_ == "cs_log") {
    lambda_.resize(num);
    for (int i = 0; i < num; ++i) {
      int label_index = i * dim + static_cast<int>(label[i]);
      Dtype per_sample_loss = 1;
      for ( int j = 0; j < dim; ++j) {
        if (j != label[i]) {
          int temp_i = i * dim + j;
          per_sample_loss += weights[temp_i] * std::min(exp(act[temp_i] - act[label_index]), (Dtype)(FLT_MAX/1e7));
        }
      }
      lambda_[i] = 1.0 / per_sample_loss;
      if(isnan(lambda_[i]) || isinf(lambda_[i])) {
        exit(0);
      }
      loss += log(std::max(per_sample_loss, Dtype(FLT_MIN)));
    }
  } else {
    LOG(INFO) << "boosting_type: " << boosting_type_ << " is unknown";
    exit(0);
  }

  loss /= n;
  // loss /= m;

  // SHOW(loss);
  // if(isinf(loss) || isnan(loss)) {
  //   for (int i = 0; i < n; ++i) {
  //     Dtype temp = 0.0;
  //     int label_index = i * m + static_cast<int>(label[i]);
  //     for ( int j = 0; j < m; ++j) {
  //       temp += exp(act[i * m + j] - act[label_index]);
  //       Dtype a = act[i * m + j] - act[label_index];
  //       SHOW(temp << " " << a);
  //     }
  //     loss += log(std::max(temp, Dtype(FLT_MIN)));
  //   }
  //   exit(0);
  // }

  if (isinf(loss) || isnan(loss)) {
    LOG(INFO) << "Exiting... because loss is inf/nan";
    for (int i = 0; i < num; ++i) {
      int label_index = i * dim + static_cast<int>(label[i]);
      Dtype per_sample_loss = 1;
      for ( int j = 0; j < dim; ++j) {
        if (j != label[i]) {
          int temp_i = i * dim + j;
          per_sample_loss += -weights[temp_i] * std::min(exp(act[temp_i] - act[label_index]), (Dtype)(FLT_MAX/1e7));
        }
      }

      if(isinf(per_sample_loss)) {
        for ( int j = 0; j < dim; ++j) {
          if (j != label[i]) {
            int temp_i = i * dim + j;
            SHOW(std::max(exp(act[temp_i] - act[label_index]), (Dtype)(FLT_MAX/1e5)));
          }
        }
        exit(0);
      }
      SHOW(lambda_[i] << " " << per_sample_loss);
    }
    exit(0);
  }

  // if (this->phase_ == TRAIN)
    // SHOW(loss);
  //   LOG(INFO) << "loss: " << loss;

  top[0]->mutable_cpu_data()[0] = loss;
  // if (this->phase_ == TRAIN)
    // SHOW(top[0]->mutable_cpu_data()[0]);
}

template <typename Dtype>
void BoostingLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(BoostingLossLayer);

}  // namespace caffe
