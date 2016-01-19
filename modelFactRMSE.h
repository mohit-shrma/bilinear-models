#ifndef _MODEL_FACT_RMSE_H_
#define _MODEL_FACT_RMSE_H_

#include "modelFactMat.h"

class ModelFactRMSE: public ModelFactMat {
  
  public:
    ModelFactRMSE(const Params &params, int nFeatures):ModelFactMat(params, nFeatures){}
    virtual void train(const Data &data, Model& bestModel);
    void computeUGrad(Eigen::MatrixXf& Ugrad, Eigen::VectorXf& iFeat, 
      Eigen::VectorXf& uFeat, float r_ui);
    void computeVGrad(Eigen::MatrixXf& Vgrad, Eigen::VectorXf& iFeat, 
      Eigen::VectorXf& uFeat, float r_ui);
    float objective(const Data& data);
};



#endif

