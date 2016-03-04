#ifndef _MODEL_RMSE_F_GRAD_H_
#define _MODEL_RMSE_F_GRAD_H_
  
#include "model.h"
#include "mathUtil.h"
#include "modelFullMat.h"
#include <chrono>

class ModelRMSEFGrad: public ModelFullMat {
  
  public:
    ModelRMSEFGrad(const Params &params, int nFeatures):ModelFullMat(params, nFeatures){}
    virtual void train(const Data &data, Model &bestModel);
    virtual float objective(const Data& data);
};

#endif
