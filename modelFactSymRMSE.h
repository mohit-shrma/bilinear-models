#ifndef _MODEL_FACT_SYM_RMSE_H_
#define _MODEL_FACT_SYM_RMSE_H_

#include "modelFactSym.h"

class ModelFactSymRMSE: public ModelFactSym {

  public:
    ModelFactSymRMSE(const Params &params, int nFeatures):ModelFactSym(params, nFeatures){}
    virtual void train(const Data& data, Model& bestModel);
    void computeUGrad(Eigen::MatrixXf& Ugrad, Eigen::VectorXf& iFeat, 
        Eigen::VectorXf& uFeat, float r_ui) ;
    void computeUSpGrad(Eigen::MatrixXf& Ugrad, const Data& data,
        float r_ui, float r_ui_est, int u, int i, Eigen::VectorXf& f_iT_U, 
        Eigen::VectorXf& f_uT_U) ;
    float objective(const Data& data);
};

#endif
