#ifndef _MODEL_BPR_F_GRAD_H_
#define _MODEL_BPR_F_GRAD_H_

#include "model.h"
#include "mathUtil.h"
#include "modelFullMat.h"
#include "util.h"
#include <array>
#include <cmath>

class ModelBPRFGrad: public ModelFullMat {
  
  public:
    ModelBPRFGrad(const Params &params, int nFeatures):ModelFullMat(params, nFeatures){}
    virtual void train(const Data &data, Model& bestModel);
};


#endif



