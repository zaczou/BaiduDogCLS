#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>
#include <boost/thread.hpp>

#include "caffe/data_transformer.hpp"
#include "caffe/util/benchmark.hpp"

#include "caffe/layers/boost_data_layer.hpp"

#define SHOW(x)   LOG(INFO) << #x << " " << x
namespace caffe {

map<std::string, caffe::BoostWeight> boostw;

template <typename Dtype>
BoostDataLayer<Dtype>::BoostDataLayer(const LayerParameter& param)
  : BaseDataLayer<Dtype>(param),
    prefetch_free_(),
    prefetch_full_(),
    reader_(param) {
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_free_.push(&prefetch_[i]);
  }

  if(param.data_param().has_boost_weight_file()) {
    boost_weight_file_ = param.data_param().boost_weight_file();
  }
}

template <typename Dtype>
BoostDataLayer<Dtype>::~BoostDataLayer() {
  this->StopInternalThread();
}

void ReadBoostWeightFile(const std::string boost_weight_file, std::map<std::string, caffe::BoostWeight>* boost_weight_map) {
  assert(boost_weight_map != NULL);
  boost_weight_map->clear();

  string db_type;
  if (boost_weight_file.find("leveldb") != std::string::npos)
    db_type = "leveldb";
  else if (boost_weight_file.find("lmdb") != std::string::npos)
    db_type = "lmdb";
  else {
    LOG(INFO) << "db type of " << boost_weight_file << " is unknown";
    exit(0);
  }

  shared_ptr<db::DB> db(db::GetDB(db_type));
  SHOW(boost_weight_file);
  db->Open(boost_weight_file, db::READ);
  shared_ptr<db::Cursor> cursor(db->NewCursor());
  while(cursor->valid()) {

    BoostWeight bw;
    bw.ParseFromString(cursor->value());

    assert(bw.w_size() == 10);
    (*boost_weight_map)[cursor->key()] = bw;
    cursor->Next();
  }
  SHOW(boost_weight_map->size());
  SHOW(boost_weight_map->begin()->first << " " << boost_weight_map->begin()->second.w_size());
  SHOW(boost_weight_map->rbegin()->first << " " << boost_weight_map->rbegin()->second.w_size());
}
template <typename Dtype>
void BoostDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  int num_classes = this->layer_param_.data_param().num_classes();
  CHECK_GE(num_classes, 1) << "num_classes should be set in the datalayer";

  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum& datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }

  // weight
  if (this->output_weights_) {
    int num_classes = this->layer_param_.data_param().num_classes();
    assert(num_classes > 0);
    vector<int> weight_shape(2);
    weight_shape[0] = batch_size;
    weight_shape[1] = num_classes;
    top[2]->Reshape(weight_shape);

    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].weight_.Reshape(weight_shape);
    }
  }

  // id
  if (this->output_ids_) {
    // we are assuming that ids are shorter than 1000 characters. Ids are image
    // paths coming from the lmdb files plus {TRAIN_, TEST_}. It sounds a fair
    // assumption. We also if this is not case for the data and stop the
    // execution.
    int max_id_size = 1000;
    vector<int> id_shape(2);
    id_shape[0] = batch_size;
    id_shape[1] = max_id_size;
    top[3]->Reshape(id_shape);

    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].id_.Reshape(id_shape);
    }
  }
}

std::string RemoveExtraStuffs(const std::string& id_string) {
  std::string res = id_string;
  while(true) {
    std::size_t pos = res.find("___");
    if(pos == std::string::npos)
      break;
    res = res.substr(0, pos) + res.substr(pos+5, res.length()-pos-5);
  }
  return res;
}

// This function is called on prefetch thread
template<typename Dtype>
void BoostDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  int num_classes = this->layer_param_.data_param().num_classes();
  CHECK_GE(num_classes, 1) << "num_classes should be set in the datalayer";

  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  Datum& datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  Dtype* top_weight = NULL;
  Dtype* top_id = NULL;

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  
  if (this->output_weights_) {
    vector<int> top_weight_shape(2);
    top_weight_shape[0] = top_shape[0];
    top_weight_shape[1] = num_classes;

    batch->weight_.Reshape(top_weight_shape);
    top_weight = batch->weight_.mutable_cpu_data();
  }

  if (this->output_ids_) {
    top_id = batch->id_.mutable_cpu_data();
  }

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    Datum& datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    // LOG(INFO) << top_data;
    this->data_transformer_->Transform(datum, &(this->transformed_data_));

    string id_string;
      if(this->phase_ == TEST)
        id_string = "TEST_";
      else if (this->phase_ == TRAIN)
        id_string = "TRAIN_";
      else {
        SHOW(this->phase_);
        exit(0);
      }
    id_string += datum.id();

    id_string = RemoveExtraStuffs(id_string);

    // Copy label.
    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }
    if (this->output_weights_) {
      if (boostw.find(id_string) == boostw.end()) {
        boostw[id_string].clear_w();
        boostw[id_string].mutable_w()->Reserve(num_classes);

        for (int j = 0; j < num_classes; ++j) {
          if (j == datum.label())
            boostw[id_string].add_w(1.0-num_classes);
          else
            boostw[id_string].add_w(1.0);
        }
      }

      for (int j = 0; j < num_classes; ++j) {
        top_weight[item_id*num_classes+j] = boostw[id_string].w(j);
      }

    }
    if (this->output_ids_) {
      assert(this->phase_ == TRAIN || this->phase_ == TEST);

      int max_id_size = 1000;
      for (int i = 0; i < max_id_size; ++i)
        top_id[item_id*max_id_size+i] = 0;
      assert(id_string.length() < max_id_size);

      for (int i = 0; i < id_string.length(); ++i)
        top_id[item_id*max_id_size+i] = Dtype(id_string[i]);

      // for (int i = 0; i < max_id_size/2; ++i)
      //   std::cout << char(top_id[item_id*max_id_size+i]) << " ";
      // std::cout << std::endl;
    }

    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<Datum*>(&datum));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
void BoostDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  output_labels_ = (top.size() >= 2);
  output_weights_ = (top.size() >= 3);
  output_ids_ = (top.size() >= 4);

  if(output_weights_ && boost_weight_file_ != "") {
    LOG(INFO) << "Reading " << boost_weight_file_ << " ...";
    ReadBoostWeightFile(boost_weight_file_, &boostw);
  }

  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_[i].data_.mutable_cpu_data();
    if (this->output_labels_) {
      prefetch_[i].label_.mutable_cpu_data();
    }
    if (this->output_weights_) {
      prefetch_[i].weight_.mutable_cpu_data();
    }
    if (this->output_ids_) {
      prefetch_[i].id_.mutable_cpu_data();
    }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < PREFETCH_COUNT; ++i) {
      prefetch_[i].data_.mutable_gpu_data();
      if (this->output_labels_) {
        prefetch_[i].label_.mutable_gpu_data();
      }
      if (this->output_weights_) {
        prefetch_[i].weight_.mutable_gpu_data();
      }
      if (this->output_ids_) {
        prefetch_[i].id_.mutable_gpu_data();
      }
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BoostDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) {
      Batch<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}

template <typename Dtype>
void BoostDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
        top[1]->mutable_cpu_data());
  }
  if (this->output_weights_) {
    // Reshape to loaded weights.
    top[2]->ReshapeLike(batch->weight_);
    // Copy the weights.
    caffe_copy(batch->weight_.count(), batch->weight_.cpu_data(),
        top[2]->mutable_cpu_data());
  }
  if (this->output_ids_) {
    // Reshape to loaded ids.
    top[3]->ReshapeLike(batch->id_);
    // Copy the ids.
    caffe_copy(batch->id_.count(), batch->id_.cpu_data(),
        top[3]->mutable_cpu_data());
  }

  prefetch_free_.push(batch);
}

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()) {
}

INSTANTIATE_CLASS(BoostDataLayer);
REGISTER_LAYER_CLASS(BoostData);

}  // namespace caffe
