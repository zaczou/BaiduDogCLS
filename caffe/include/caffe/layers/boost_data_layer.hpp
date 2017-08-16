#ifndef CAFFE_BOOST_DATA_LAYER_HPP_
#define CAFFE_BOOST_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class BoostDataLayer :
   public BaseDataLayer<Dtype> , public InternalThread {
 public:
  explicit BoostDataLayer(const LayerParameter& param);
  virtual ~BoostDataLayer();

  // Brought over from BasePrefetchingDataLayer
  // This method may not be overridden.
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Prefetches batches (asynchronously if to GPU memory)
  static const int PREFETCH_COUNT = 3;

  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // BoostDataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "BoostData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 4; }

 protected:
  // Brought over from BasePrefetchingDataLayer
  virtual void InternalThreadEntry();
  void load_batch(Batch<Dtype>* batch);

  Batch<Dtype> prefetch_[PREFETCH_COUNT];
  BlockingQueue<Batch<Dtype>*> prefetch_free_;
  BlockingQueue<Batch<Dtype>*> prefetch_full_;

  Blob<Dtype> transformed_data_;
  std::string boost_weight_file_;

  DataReader reader_;

  bool output_labels_;
  bool output_weights_;
  bool output_ids_;
};

}  // namespace caffe

#endif  // CAFFE_BOOST_DATA_LAYER_HPP_
