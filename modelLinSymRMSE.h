#ifndef _MODEL_LIN_SYM_RMSE_H_
#define _MODEL_LIN_SYM_RMSE_H_

#include "modelLinSym.h"

class ModelLinSymRMSE:public ModelLinSym {
  public:
    ModelLinSymRMSE(const Params &params, int nFeatures):ModelLinSym(params, nFeatures){}
    virtual void train(const Data &data, Model& bestModel);
    virtual float objective(const Data& data);
    void computeUGrad(Eigen::MatrixXf& Ugrad, Eigen::VectorXf& iFeat, 
      Eigen::VectorXf& uFeat, float r_ui);
    void computewGrad(Eigen::VectorXf& wgrad, Eigen::VectorXf& iFeat,  
      Eigen::VectorXf& uFeat, float r_ui);
};

#endif

