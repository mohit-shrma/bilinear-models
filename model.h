#ifndef _MODEL_H_
#define _MODEL_H_

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include "const.h"
#include "datastruct.h"
#include "model.h"
#include "GKlib.h"


class Model {
  
  public:
    int nFeatures;
    Eigen::MatrixXf W;
    
    float l2Reg;
    float nucReg;
    float learnRate;
    int rank;
    int maxIter;
    
    Model(const Params &params, int p_nFeatures);

    virtual float objective(const Data& data) {
      std::cerr << "\nBase objective method" << std::endl;
      return -1;
    };

    virtual void train(const Data &data, Model& bestModel) {
      std::cerr << "\nTraining not in base class";
    };
    
    float computeRecall(gk_csr_t *mat, const Data &data, int N, 
        std::unordered_set<int> items);
    float computeRMSE(gk_csr_t *mat, const Data& data);
    bool isTerminateModel(Model& bestModel, const Data& data, int iter,
      int& bestIter, float& bestRecall, float& prevRecall); 

    bool isTerminateModelObj(Model& bestModel, const Data& data, int iter,
      int& bestIter, float& bestObj, float& prevObj);

};

#endif


