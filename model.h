#ifndef _MODEL_H_
#define _MODEL_H_

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <thread>
#include "mathUtil.h"
#include "const.h"
#include "datastruct.h"
#include "GKlib.h"
#include "io.h"

class Model {
  
  public:
    Eigen::MatrixXf W;
    Eigen::MatrixXf U;
    Eigen::MatrixXf V;
    Eigen::VectorXf w;
    int nFeatures;
    
    float l2Reg;
    float wReg;
    float nucReg;
    float learnRate;
    float pcSamples;
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
    
    virtual float estPosRating(int u, int item, const Data& data,
      Eigen::VectorXf& pdt) = 0;
    virtual float estNegRating(int u, int item, const Data& data, 
      Eigen::VectorXf& pdt) = 0;
    float computeRecall(gk_csr_t *mat, const Data &data, int N, 
        std::unordered_set<int> items);
    float computeRecallPar(gk_csr_t *mat, const Data &data, int N, 
        std::unordered_set<int> items);
    float computeRMSE(gk_csr_t *mat, const Data& data);
    bool isTerminateModel(Model& bestModel, const Data& data, int iter,
      int& bestIter, float& bestRecall, float& prevRecall); 
    bool isTerminateModelObj(Model& bestModel, const Data& data, int iter,
      int& bestIter, float& bestObj, float& prevObj);
    void computeRecallUsers(gk_csr_t *mat, int uStart, int uEnd, 
      const Data& data, int N, std::unordered_set<int>& items, 
      std::vector<bool>& isTestUser, std::vector<float>& uRecalls);

};

#endif


