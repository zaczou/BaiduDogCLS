#include <vector>
#include <string>

#include "caffe/util/db.hpp"
#include "boost/scoped_ptr.hpp"

#include "caffe/sgd_solvers.hpp"

namespace caffe {
#define SHOW(x) LOG(INFO) << #x << " = " << x

extern map<std::string, BoostWeight> boostw;

template <typename Dtype>
int BoostSolver<Dtype>::FindBlobIndex(const shared_ptr<Net<Dtype> >& net, const std::string& blob_name) {
  int target_layer_index = -1;

  for (int i = 0; i < net->blobs().size(); ++i) {
    if(net->blob_names()[i] == blob_name) {
      LOG(INFO) << "matched: " << "blob#" << i << " (" << net->blob_names()[i] << ") " << blob_name;
      target_layer_index = i;
    }
  }

  assert(target_layer_index >= 0);
  return target_layer_index;
}


// We drop 0.5 in calculation of gradient
template <typename Dtype>
double BoostSolver<Dtype>::CalculateAlphaGradient(const std::map<std::string, BoostWeight>& boostw,
                                                  const std::map<std::string, BoostWeight>& activations,
                                                  const std::map<std::string, int>& label_map,
                                                  double alpha) {
  int num_train = 0;
  double gradient = 0;
  for(std::map<std::string, int>::const_iterator it = label_map.begin(); it != label_map.end(); ++it) {
    assert(it->first.length() > 5);
    if(it->first.compare(0, 5, "TEST") == 0) continue;
    ++ num_train;


    string id_string = it->first;
    if(boostw.find(id_string) == boostw.end()) {SHOW(id_string); exit(0);}
    if(activations.find(id_string) == activations.end()) {SHOW(id_string); exit(0);}
    int label = it->second;
    const BoostWeight& old_weight = boostw.find(id_string)->second;
    const BoostWeight& act = activations.find(id_string)->second;
    int num_classes = act.w_size();

    for (int j = 0; j < num_classes; ++j)
      if(isnan(act.w(j)) || isinf(act.w(j))) {
        SHOW("act is nan or inf");
        SHOW(act.w(j));
        SHOW(label);
        exit(0);
      }

    for (int j = 0; j < num_classes; ++j)
      if (j != label) {
        gradient += -old_weight.w(j) * (act.w(label) - act.w(j)) * exp(std::min(double(10), -0.5*alpha*(act.w(label) - act.w(j)))) / num_classes;
      }

    if (isinf(gradient) || isnan(gradient)) {
      LOG(INFO) << "gradient became nan or inf";

      SHOW(num_classes);
      for (int j = 0; j < num_classes; ++j)
        if (j != label) {
          LOG(INFO) << "===================================================================";
          SHOW(alpha);
          SHOW(label);
          SHOW(j << " " << old_weight.w(j) * (act.w(label) - act.w(j)) * exp(-0.5*alpha*(act.w(label) - act.w(j))));
          SHOW(old_weight.w(j));
          SHOW((act.w(label) - act.w(j)));
        }

      exit(0);
    }

  }
  return gradient / num_train;
}

template <typename Dtype>
void BoostSolver<Dtype>::CalculateActivations(std::map<std::string, BoostWeight>* activations,
                                         std::map<std::string, int>* label_map,
                                         bool if_train,
                                         bool if_test,
                                         const std::string& blob_name,
                                         int limit) {
  std::vector< shared_ptr<Net<Dtype> > > nets;
  nets.push_back(Solver<Dtype>::net_);
  nets.push_back(Solver<Dtype>::test_nets_[0]);

  vector<bool> ifs;
  ifs.push_back(if_train);
  ifs.push_back(if_test);

  assert(nets.size() == ifs.size());

  CHECK_NOTNULL(nets[1].get())->ShareTrainedLayersWith(nets[0].get());

  for (int net_iter = 0; net_iter < nets.size(); ++net_iter) {
    if(!ifs[net_iter])
      continue;

    shared_ptr<Net<Dtype> > net = nets[net_iter];

    int target_layer_index = FindBlobIndex(net, blob_name);

    SHOW(target_layer_index);
    SHOW(net->blobs().size());

    assert(target_layer_index >= 0);
    assert(target_layer_index < net->blobs().size());

    int batch_size;
    int duplicate_count = 0;

    do {
      net->Forward();

      const Blob<Dtype>& act = *nets[net_iter]->blobs()[target_layer_index];
      batch_size = act.shape()[0];
      int feature_dim = act.count() / batch_size;

      const Blob<Dtype>& label = *(nets[net_iter]->blobs()[1]);
      //const Blob<Dtype>& weight = *(nets[net_iter]->blobs()[2]);
      const Blob<Dtype>& id = *(nets[net_iter]->blobs()[3]);

      const Dtype* id_data = id.cpu_data();
      const Dtype* label_data = label.cpu_data();
      const Dtype* act_data = act.cpu_data();

      assert(id_data && label_data && act_data);

      int max_id_size = id.shape()[1];

      bool ok = false;
      for(int i = 0; i < batch_size; ++i) {
        std::string id_string;
        for (int j = 0; j < max_id_size; ++j) {
          char new_char = (char)(id_data[i*max_id_size + j]);
          if (new_char == 0)
            break;
          id_string += new_char;
        }
        if (activations->find(id_string) == activations->end()) {
          if(activations->size() % 100 == 0)
            SHOW(activations->size());

          if(limit > 0 && activations->size() >= limit) {
            ok = true;
            break;
          }
          (*label_map)[id_string] = label_data[i];
          (*activations)[id_string] = BoostWeight();
          BoostWeight& new_weight = (*activations)[id_string];
          new_weight.mutable_w()->Reserve(feature_dim);
          for(int k = 0; k < feature_dim; ++k) {
            new_weight.add_w(act_data[feature_dim*i+k]);
          }
        } else {
          ++ duplicate_count;
        }
      }
      if(ok)break;
    } while(duplicate_count < batch_size);
  }
}



template<typename Dtype>
double BoostSolver<Dtype>::CalculateAccuracy(
                              std::map<std::string, BoostWeight>& activations,
                              std::map<std::string, int>& label_map) {
  LOG(INFO) << "in Calculating Accuracy";
  if(activations.size() == 0) {
    LOG(INFO) << "Calculating Activations";
    CalculateActivations(&activations, &label_map, true, true, "fc8ft");
  }
  assert(activations.size() == label_map.size());

  LOG(INFO) << "Calculating Alpha";
  double alpha = CalculateAlphaFast(boostw, activations, label_map, true);

  SHOW(alpha);

  int num_correct_train = 0;
  int num_total_train = 0;

  int num_correct_test = 0;
  int num_total_test = 0;

  for(std::map<std::string, int>::const_iterator it = label_map.begin(); it != label_map.end(); ++it) {
    assert(it->first.length() >= 5);

    string id_string = it->first;
    int label = it->second;

    assert(boostw.find(id_string) != boostw.end());
    assert(activations.find(id_string) != boostw.end());

    const BoostWeight& old_weight = boostw.find(id_string)->second;
    const BoostWeight& act = activations.find(id_string)->second;
    int num_classes = act.w_size();

    assert(num_classes > 0);

    int l = label;
    bool correct = true;

    for(int k = 0; k < num_classes; ++k) {
      if (k != l && (log(old_weight.w(k)) - 0.5*alpha*(act.w(l)-act.w(k))) > 1e-10) {
        correct = false;
        break;
      }
    }

    if(it->first.compare(0, 5, "TRAIN") == 0) {
      if(correct)
        ++num_correct_train;
      ++num_total_train;
    } else if(it->first.compare(0, 4, "TEST") == 0) {
      if(correct)
        ++num_correct_test;
      ++num_total_test;
    }
  }

  double train_loss = CalculateBoostingLoss(boostw, activations, label_map, alpha, true, false) / num_total_train;
  double test_loss = CalculateBoostingLoss(boostw, activations, label_map, alpha, false, true) / num_total_test;

  SHOW(train_loss);
  SHOW(test_loss);

  SHOW(num_correct_train);
  SHOW(num_total_train);

  SHOW(num_correct_test);
  SHOW(num_total_test);

  double train_accuracy = double(num_correct_train) / num_total_train;
  double test_accuracy = double(num_correct_test) / num_total_test;

  SHOW(train_accuracy);
  SHOW(test_accuracy);

  return test_accuracy;
}


template <typename Dtype>
double BoostSolver<Dtype>::CalculateAlphaFast(const std::map<std::string, BoostWeight>& boostw,
                                              const std::map<std::string, BoostWeight>& activations,
                                              const std::map<std::string, int>& label_map,
                                              bool verbose) {
  int max_iter = 10;
  double alpha_min = 0;
  double alpha_max = 1;

  
  double grad_max = CalculateAlphaGradient(boostw, activations, label_map, alpha_max);
  double grad_min = CalculateAlphaGradient(boostw, activations, label_map, alpha_min);

  if(grad_min > 0) {
    LOG(INFO) << "The aux classifier has not been properly trained (so far) " 
              << "alpha_min: " << alpha_min
              << "grad_min: " << grad_min
              << "alpha_max: " << alpha_max
              << "grad_max: " << grad_max;
  }

  while(isinf(grad_min) || isnan(grad_min)) {
    LOG(INFO) << "grad_min is nan|inf. Thus we divide alpha_min by 2. old alpha_min" << alpha_min << " alpha_min is now " << alpha_min/2;
    alpha_min /= 2;
    grad_min = CalculateAlphaGradient(boostw, activations, label_map, alpha_min);
  }

  while (grad_max < 0) {
    LOG(INFO) << "grad_max is negative Thus we multiply alpha_max by 2. old alpha_max" << alpha_max << " alpha_max is now " << alpha_max*2;

    alpha_min = alpha_max;
    grad_min = grad_max;

    alpha_max *= 2;
    grad_max = CalculateAlphaGradient(boostw, activations, label_map, alpha_max);
  }

  if (verbose) {
    SHOW(alpha_min << " " << grad_min);
    SHOW(alpha_max << " " << grad_max);
  }

  for(int iter = 0; iter < max_iter; ++iter) {
    double alpha_mid = (alpha_min + alpha_max) / 2;
    double grad_mid = CalculateAlphaGradient(boostw, activations, label_map, alpha_mid);

    if(verbose)
      SHOW(alpha_mid << " " << grad_mid);

    if (grad_mid > 0)
      alpha_max = alpha_mid;
    else
      alpha_min = alpha_mid;
  }
  return (alpha_min+alpha_max) / 2;
}

template <typename Dtype>
double BoostSolver<Dtype>::CalculateBoostingLoss(const std::map<std::string, BoostWeight>& boostw,
                                                 const std::map<std::string, BoostWeight>& activations,
                                                 const std::map<std::string, int>& label_map,
                                                 double alpha,
                                                 bool if_train,
                                                 bool if_test) {
    double loss = 0;
    for(std::map<std::string, int>::const_iterator it = label_map.begin(); it != label_map.end(); ++it) {
      if((if_test && it->first.compare(0, 4, "TEST") == 0)
      || (if_train && it->first.compare(0, 5, "TRAIN") == 0)) {

        string id_string = it->first;
        int label = it->second;
        const BoostWeight& old_weight = boostw.find(id_string)->second;
        const BoostWeight& act = activations.find(id_string)->second;
        int num_classes = act.w_size();

        for (int j = 0; j < num_classes; ++j)
          if (j != label) {
            // The std::min is just a hack
            loss += old_weight.w(j) * exp(std::min(double(100), -0.5*alpha*(act.w(label) - act.w(j))));
          }

        if(isnan(loss)) {
          LOG(INFO) << "loss became nan";
          SHOW(alpha);
          for (int j = 0; j < num_classes; ++j)
            if (j != label)
              SHOW(j << " " << -old_weight.w(j) * exp(-0.5*alpha*(act.w(label) - act.w(j))));
          exit(0);
        }
      }
    }

  return loss;
}


template <typename Dtype>
void BoostSolver<Dtype>::CalculateBoostWeights(const std::map<std::string, BoostWeight>& boostw,
                                               const std::map<std::string, BoostWeight>& activations,  
                                               const std::map<std::string, int> label_map,
                                               double alpha,
                                               std::map<std::string, BoostWeight>* output_boostw) {
  if (output_boostw == NULL) {
    LOG(ERROR) << "output_boostw is NULL";
    return;
  }
  output_boostw->clear();
  // TODO(mmoghimi)
  // net_->Forward();
  LOG(INFO) << "Calculating Boosting Weights for " << Solver<Dtype>::net_->name();

  for(std::map<std::string, int>::const_iterator it = label_map.begin(); it != label_map.end(); ++it) {
    string id_string = it->first;
    int label = it->second;
    assert(boostw.find(id_string) != boostw.end());
    assert(activations.find(id_string) != boostw.end());

    const BoostWeight& old_weight = boostw.find(id_string)->second;
    const BoostWeight& act = activations.find(id_string)->second;
    int num_classes = act.w_size();

    assert(num_classes > 0);

    (*output_boostw)[id_string] = BoostWeight();
    BoostWeight& new_weight = (*output_boostw)[id_string];
    new_weight.mutable_w()->Reserve(num_classes);
    Dtype w_il = 0;
    int l = label;

    for(int k = 0; k < num_classes; ++k) {
      Dtype new_w = 0;
      if (k != l) {
        new_w = old_weight.w(k) * exp(-0.5*alpha*(act.w(l)-act.w(k)));
        w_il -= new_w;
      }
      new_weight.add_w(new_w);
    }
    new_weight.set_w(l, w_il);
  }

}

template <typename Dtype>
void BoostSolver<Dtype>::Solve(const char* resume_file) {
  LOG(INFO) << "Running BoostSolver";
  map<std::string, BoostWeight> activations;
  map<std::string, int> label_map;

  Solver<Dtype>::Solve(resume_file);

  CalculateActivations(&activations, &label_map, true, false, "fc8ft");
  CalculateActivations(&activations, &label_map, false, true, "fc8ft");
  CalculateAccuracy(activations, label_map);

  std::string output_boostweights = Solver<Dtype>::param_.output_boost_weight_file();

  if (output_boostweights != "") {
    map<std::string, BoostWeight> output_boostw;

    LOG(INFO) << "Calculating alpha...";
    // Even though we are passing all the activations (train and test), alpha is calculated only on the training subset
    double alpha = CalculateAlphaFast(boostw, activations, label_map, true);
    double shrinkage = .1;
    alpha *= shrinkage;
    LOG(INFO) << "Calculating boosting weights...";
    CalculateBoostWeights(boostw, activations, label_map, alpha, &output_boostw);
    LOG(INFO) << "Writing new weights to " << output_boostweights << "...";
    WriteBoostWeightFile(output_boostweights, output_boostw);
  }
}

template <typename Dtype>
void BoostSolver<Dtype>::WriteBoostWeightFile(
                          const std::string& output_boostweights_filename,
                          const std::map<std::string, BoostWeight>& output_boostw) {
  string db_type = output_boostweights_filename.substr(output_boostweights_filename.rfind("_")+1, output_boostweights_filename.length() - output_boostweights_filename.rfind("_"));
  LOG(INFO) << "output_boostweights_filename:" << output_boostweights_filename;
  if (db_type == "") {
  	LOG(INFO) << "Could not figure out the DB type based on file name. Choosing lmdb.";
  	db_type = "lmdb";
  }
  LOG(INFO) << db_type;

  boost::scoped_ptr<db::DB> db(db::GetDB(db_type));
  db->Open(output_boostweights_filename, db::NEW);
  boost::scoped_ptr<db::Transaction> txn(db->NewTransaction());
  int count = 0;
  for(std::map<std::string, BoostWeight>::const_iterator iter = output_boostw.begin();
      iter != output_boostw.end();
      ++iter) {
    const std::string& id = iter->first;
    const BoostWeight& bw = iter->second;

    std::string weight_buffer;
    bw.SerializeToString(&weight_buffer);

    txn->Put(id, weight_buffer);
    ++ count;

    if (count % 1000 == 0) {
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(ERROR) << count << " weight written.";
    }
  }
  if (count % 1000 != 0)
    txn->Commit();
  db->Close();
}

INSTANTIATE_CLASS(BoostSolver);
REGISTER_SOLVER_CLASS(Boost);

}  // namespace caffe
