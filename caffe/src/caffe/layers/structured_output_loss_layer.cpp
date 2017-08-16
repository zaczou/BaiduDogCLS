/*
 * structured_output_loss_layer.cpp
 *
 *  Created on: Jun 9, 2017
 *      Author: lab-xiong.jiangfeng
 */

#include <algorithm>
#include <vector>

#include "caffe/layers/structured_output_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void StructuredOutputLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();//score map
  const Dtype* map = bottom[1]->cpu_data();//overlap or value of gaussian
  const Dtype* instanceWeigth = bottom[2]->cpu_data();


  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  vector<int> max_conf_index_shape;
  max_conf_index_shape.push_back(num);
  max_conf_index_.Reshape(max_conf_index_shape);
  signed_slack_variable_.ReshapeLike(*bottom[0]);

  int* max_conf_index_data = max_conf_index_.mutable_cpu_data();
  Dtype* signed_slack_variable_data = signed_slack_variable_.mutable_cpu_data();
  Dtype* loss = top[0]->mutable_cpu_data();
  loss[0]=0;



  for (int i = 0; i < num; ++i) {
	  //get the index of target location for each example
	  std::vector<std::pair<float,int> > pairs;
	  for(int k=0;k<dim;++k){
		  pairs.push_back(std::make_pair(map[i*dim +k],k));
	  }
	  //find argmax{score}
	  std::sort(pairs.begin(),pairs.end(),std::greater<std::pair<Dtype,int> >());
	  max_conf_index_data[i] = pairs[0].second;
	  Dtype f_max = bottom_data[i*dim +pairs[0].second];

    for (int j = 0; j < dim; ++j) {
      signed_slack_variable_data[i * dim + j] = ((Dtype(1.0) - map[i * dim + j]) - (f_max - bottom_data[i * dim + j]));
      signed_slack_variable_data[i * dim + j] = std::max(Dtype(0), signed_slack_variable_data[i * dim + j]);

      //L2
      if(this->layer_param_.structured_output_param().norm()== StructuredOutputLossParameter_Norm_L2 ){
          signed_slack_variable_data[i * dim + j] = std::max(Dtype(0), signed_slack_variable_data[i * dim + j])*std::sqrt(instanceWeigth[i * dim + j]);
      }
      else if(this->layer_param_.structured_output_param().norm()== StructuredOutputLossParameter_Norm_MIX){
		  signed_slack_variable_data[i * dim + j] = std::max(Dtype(0), signed_slack_variable_data[i * dim + j]);
		  if(signed_slack_variable_data[i * dim + j]>1){
			  loss[0]+=Dtype(1.0/num)*(signed_slack_variable_data[i * dim + j] - 0.5)*instanceWeigth[i * dim + j];
		  }
		  else{
			  loss[0]+=Dtype(0.5/num)*signed_slack_variable_data[i * dim + j]*signed_slack_variable_data[i * dim + j]*instanceWeigth[i * dim + j];
		  }
      }

      //std::cout<<"signed_slack_variable_data[0,"<<j<<"]"<<signed_slack_variable_data[i * dim + j]<<std::endl;
    }

  }

  //L1
  if(this->layer_param_.structured_output_param().norm()== StructuredOutputLossParameter_Norm_L1){
	  caffe_mul(count,signed_slack_variable_data,instanceWeigth,signed_slack_variable_data);
	  loss[0] = Dtype(1.0/num)* caffe_cpu_asum(count,signed_slack_variable_data);
  }
  else if(this->layer_param_.structured_output_param().norm()== StructuredOutputLossParameter_Norm_L2){
	  //L2
	  loss[0] = Dtype(0.5/num)* caffe_cpu_dot(count,signed_slack_variable_data,signed_slack_variable_data);
  }
}

template <typename Dtype>
void StructuredOutputLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to instance_weights";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int* max_conf_index_data = max_conf_index_.cpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;
    const Dtype* instanceWeigth = bottom[2]->cpu_data();
    const Dtype* signed_slack_variable_data = signed_slack_variable_.cpu_data();

    caffe_set(count,Dtype(0),bottom_diff);
    const Dtype loss_weight = top[0]->cpu_diff()[0];

    //we have already save slack variable in bottom_diff
    for (int i = 0; i < num; ++i) {
        for (int j = 0; j < dim; ++j) {
        	if(j==max_conf_index_data[i]){
        		for(int k=0;k<dim;++k){
        			if(k==j) continue;
        			if(this->layer_param_.structured_output_param().norm()== StructuredOutputLossParameter_Norm_L1){
        				//L1
        				bottom_diff[i*dim +j]+= -1*(signed_slack_variable_data[i*dim+k]>0)*instanceWeigth[i*dim+k];
        			}
        			else if(this->layer_param_.structured_output_param().norm()== StructuredOutputLossParameter_Norm_L2){
        			//L2
        			bottom_diff[i*dim +j]+= -1*(signed_slack_variable_data[i*dim+k])*std::sqrt(instanceWeigth[i*dim+k]);
        			}

        			else if(this->layer_param_.structured_output_param().norm()== StructuredOutputLossParameter_Norm_MIX){
						if(signed_slack_variable_data[i * dim + k]>1){
							bottom_diff[i*dim +j]+= Dtype(-1.0)*(signed_slack_variable_data[i*dim+k]>0)*instanceWeigth[i*dim+k];
						}
						else{
							bottom_diff[i*dim +j]+= Dtype(-1.0)*(signed_slack_variable_data[i*dim+k])*instanceWeigth[i*dim+k];
						}
        			}
        		}
        	}
        	else{
        			if(this->layer_param_.structured_output_param().norm()== StructuredOutputLossParameter_Norm_L1){
        			//L1
        				bottom_diff[i*dim +j] = (signed_slack_variable_data[i*dim+j]>0)*instanceWeigth[i*dim+j];
        			}
        			else if(this->layer_param_.structured_output_param().norm()== StructuredOutputLossParameter_Norm_L2){
        			//L2
        				bottom_diff[i*dim +j] = signed_slack_variable_data[i*dim+j]*std::sqrt(instanceWeigth[i*dim+j]);
        			}
        			else if(this->layer_param_.structured_output_param().norm()== StructuredOutputLossParameter_Norm_MIX){

						//L1-smooth
						if(signed_slack_variable_data[i * dim + j]>1){
							bottom_diff[i*dim +j] = (signed_slack_variable_data[i*dim+j]>0)*instanceWeigth[i*dim+j];
						}
						else{
							bottom_diff[i*dim +j] = (signed_slack_variable_data[i*dim+j])*instanceWeigth[i*dim+j];
						}
        			}
        	}
        	bottom_diff[i * dim + j] = bottom_diff[i * dim + j] * loss_weight/num;
          //std::cout<<max_conf_index_data[i]<<" "<<j<<" bottom_diff[i,j]= "<<bottom_diff[i * dim + j]<<std::endl;
        }
    }
   }
  }
INSTANTIATE_CLASS(StructuredOutputLossLayer);
REGISTER_LAYER_CLASS(StructuredOutputLoss);
}  // namespace caffe
