#ifndef _MODEL_LINEAR_RMSE_H_
#define _MODEL_LINEAR_RMSE_H_
#include "modelLinear.h"
class ModelLinearRMSE: public ModelLinear {
  public:
    ModelLinearRMSE(const Params &params, int nFeatures):ModelLinear(params, nFeatures){}
    virtual void train(const Data &data, Model& bestModel);
    virtual float objective(const Data& data);
};

#endif

