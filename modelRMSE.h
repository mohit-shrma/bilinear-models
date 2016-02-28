#ifndef _MODEL_RMSE_H_
#define _MODEL_RMSE_H_
  
#include "model.h"
#include "mathUtil.h"
#include "modelFullMat.h"
#include <chrono>

class ModelRMSE: public ModelFullMat {
  
  public:
    ModelRMSE(const Params &params, int nFeatures):ModelFullMat(params, nFeatures){}
    virtual void train(const Data &data, Model &bestModel);
    virtual float objective(const Data& data);
    void computeRMSEGrad(Eigen::VectorXf& uFeat, Eigen::VectorXf& iFeat,
        Eigen::MatrixXf& Wgrad, float r_ui);
  void computeRMSESparseGrad(int u, int i, float r_ui, 
      Eigen::MatrixXf& Wgrad, Eigen::VectorXf& pdt, const Data& data);
    void gradCheck(Eigen::VectorXf& uFeat, Eigen::VectorXf& iFeat,
        Eigen::MatrixXf& Wgrad, float r_ui);
};

#endif
