#ifndef _MODEL_RMSE_H_
#define _MODEL_RMSE_H_
  
#include "model.h"

class ModelRMSE: public Model {
  
  public:
    ModelRMSE(const Params &params, int nFeatures):Model(params, nFeatures){}
    virtual void train(const Data &data, Model &bestModel);
    void computeGrad(Eigen::VectorXf& uFeat, Eigen::VectorXf& iFeat,
        Eigen::MatrixXf& Wgrad, float r_ui);
};

#endif
