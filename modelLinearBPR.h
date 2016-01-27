#ifndef _MODEL_LINEAR_BPR_H_
#define _MODEL_LINEAR_BPR_H_
#include "modelLinear.h"
class ModelLinearBPR: public ModelLinear {

  public:
    ModelLinearBPR(const Params &params, int nFeatures):ModelLinear(params, nFeatures){}
    virtual void train(const Data &data, Model& bestModel);
    void computewGrad(int u, int i, int j, const Data& data, 
      Eigen::VectorXf& wgrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat, 
      Eigen::VectorXf& uFeat);
};


#endif


