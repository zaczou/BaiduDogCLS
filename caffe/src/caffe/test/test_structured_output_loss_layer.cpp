/*
 * test_structured_output_loss_layer.cpp
 *
 *  Created on: Jun 11, 2017
 *      Author: lab-xiong.jiangfeng
 */

#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/structured_output_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class StructuredOutputLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  StructuredOutputLossLayerTest()
      : blob_bottom_score_(new Blob<Dtype>(3, 1, 2, 2)),
        blob_bottom_map_(new Blob<Dtype>(3, 1, 2, 2)),
        blob_bottom_instanceweight_(new Blob<Dtype>(3, 1, 2, 2)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);

    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_score_);
    blob_bottom_vec_.push_back(blob_bottom_score_);


    Dtype* blob_bottom_map_data = blob_bottom_map_->mutable_cpu_data();
    //gasussian label

    for(int n=0;n<3;n++){
    	Dtype max_x = 0;
		for(int y=0;y < 2;y++){
			for (int x = 0; x < 2; x++) {
				blob_bottom_map_data[n*4+2*y+x] = std::exp(-(pow(x,2)+std::pow(y,2))/(2.0));
				if(max_x<blob_bottom_map_data[n*4+2*y+x]){
					max_x  = blob_bottom_map_data[n*4+2*y+x];
				}
			}
		}
		for(int d=0;d<4;d++){
			blob_bottom_map_data[n*4+d] = blob_bottom_map_data[n*4+d]/max_x;
		}
    }

    blob_bottom_vec_.push_back(blob_bottom_map_);

    FillerParameter filler_param_2;
    filler_param_2.set_std(1.0);
    PositiveUnitballFiller<Dtype> filler2(filler_param_2);
    filler2.Fill(this->blob_bottom_instanceweight_);

    blob_bottom_vec_.push_back(blob_bottom_instanceweight_);

    blob_top_vec_.push_back(blob_top_loss_);

  }
  virtual ~StructuredOutputLossLayerTest() {
    delete blob_bottom_score_;
    delete blob_bottom_map_;
    delete blob_bottom_instanceweight_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_score_;
  Blob<Dtype>* const blob_bottom_map_;
  Blob<Dtype>* const blob_bottom_instanceweight_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(StructuredOutputLossLayerTest, TestDtypesAndDevices);


TYPED_TEST(StructuredOutputLossLayerTest, TestGradientMIX) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  StructuredOutputLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}
TYPED_TEST(StructuredOutputLossLayerTest, TestGradientL1) {
	typedef typename TypeParam::Dtype Dtype;
	  LayerParameter layer_param;
	  StructuredOutputLossParameter* sol_param = layer_param.mutable_structured_output_param();
	  sol_param->set_norm(StructuredOutputLossParameter_Norm_L1);
	  StructuredOutputLossLayer<Dtype> layer(layer_param);

	  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
	  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
	      this->blob_top_vec_, 0);
}
TYPED_TEST(StructuredOutputLossLayerTest, TestGradientL2) {
	typedef typename TypeParam::Dtype Dtype;
	  LayerParameter layer_param;
	  StructuredOutputLossParameter* sol_param = layer_param.mutable_structured_output_param();
	  sol_param->set_norm(StructuredOutputLossParameter_Norm_L2);
	  StructuredOutputLossLayer<Dtype> layer(layer_param);

	  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
	  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
	      this->blob_top_vec_, 0);
}

}  // namespace caffe


