#ifndef _MODEL_FACT_RMSE_H_
#define _MODEL_FACT_RMSE_H_

#include "modelFactMat.h"

class ModelFactRMSE: public ModelFactMat {
  
  public:
    ModelFactRMSE(const Params &params, int nFeatures):ModelFactMat(params, nFeatures){}
    virtual void train(const Data &data, Model& bestModel);
    void computeUGrad(Eigen::MatrixXf& Ugrad, Eigen::VectorXf& iFeat, 
      Eigen::VectorXf& uFeat, float r_ui);
    void computeUSpGrad(Eigen::MatrixXf& Ugrad, float r_ui, 
      float r_ui_est, Eigen::VectorXf& f_u_f_i_diff, Eigen::VectorXf& f_iT_V);
    void computeVGrad(Eigen::MatrixXf& Vgrad, Eigen::VectorXf& iFeat, 
      Eigen::VectorXf& uFeat, float r_ui);
    void computeVSpGrad(Eigen::MatrixXf& Vgrad, const Data& data, 
      int item, float r_ui, float r_ui_est,  
      Eigen::VectorXf& f_u_f_i_T_U);
    float objective(const Data& data);
};



#endif

