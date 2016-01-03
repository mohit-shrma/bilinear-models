#ifndef _MODEL_RMSE_H_
#define _MODEL_RMSE_H_
  
#include "model.h"
#include "mathUtil.h"

class ModelRMSE: public Model {
  
  public:
    ModelRMSE(const Params &params, int nFeatures):Model(params, nFeatures){}
    virtual void train(const Data &data, Model &bestModel);
    virtual float objective(const Data& data);
    void computeGrad(Eigen::VectorXf& uFeat, Eigen::VectorXf& iFeat,
        Eigen::MatrixXf& Wgrad, float r_ui);
};

#endif
