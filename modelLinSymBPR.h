#ifndef _MODEL_LIN_SYM_BPR_H_
#define _MODEL_LIN_SYM_BPR_H_
#include "modelLinSym.h"
class ModelLinSymBPR: public ModelLinSym {
  public:
    ModelLinSymBPR(const Params &params, int nFeatures):ModelLinSym(params, nFeatures){}
    virtual void train(const Data &data, Model& bestModel);
    void computeUGrad(int u, int i, int j, const Data& data, 
      Eigen::MatrixXf& Ugrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat,
      Eigen::VectorXf& uFeat);
    void computewGrad(int u, int i, int j, const Data& data, 
      Eigen::VectorXf& wgrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat, 
      Eigen::VectorXf& uFeat);
};


#endif


