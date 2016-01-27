#ifndef _MODEL_LIN_FACT_MAT_RMSE_H_
#define _MODEL_LIN_FACT_MAT_RMSE_H_

#include "modelLinFactMat.h"

class ModelLinFactMatRMSE: public ModelLinFactMat {
  
  public:
    ModelLinFactMatRMSE(const Params& params, int nFeatures):ModelLinFactMat(params, nFeatures){}
    virtual void train(const Data &data, Model& bestModel);
    void computeUGrad(Eigen::MatrixXf& Ugrad, Eigen::VectorXf& iFeat,
      Eigen::VectorXf& uFeat, float r_ui);
    void computeVGrad(Eigen::MatrixXf& Vgrad, Eigen::VectorXf& iFeat, 
      Eigen::VectorXf& uFeat, float r_ui);
    void computewGrad( Eigen::VectorXf& wgrad, Eigen::VectorXf& iFeat, 
      Eigen::VectorXf& uFeat, float r_ui);
    virtual float objective(const Data& data);
};

#endif


