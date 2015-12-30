#ifndef _MODEL_H_
#define _MODEL_H_

#include <Eigen/Dense>
#include "const.h"

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

    virtual float objective(const Data& data); 
    virtual void train(const Data &data, Model& bestModel);
    
    float computeRecall(gk_csr_t *mat, const Data &data, int N, 
        std::unordered_set<int> items);

    bool isTerminateModel(Model& bestModel, const Data& data, int iter,
      int& bestIter, float& bestRecall, float& prevRecall); 

};

#endif


