#ifndef _MODEL_FACT_RMSE_H_
#define _MODEL_FACT_RMSE_H_

#include "modelFactMat.h"

class ModelFactRMSE: public ModelFactMat {
  
  public:
    ModelFactRMSE(const Params &params, int nFeatures):ModelFactMat(params, nFeatures){}
    virtual void train()(const Data &data, Model& bestModel);
}



#endif

