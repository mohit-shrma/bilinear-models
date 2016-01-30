#ifndef _MODEL_FULL_HINGE_H
#define _MODEL_FULL_HINGE_H

#include "model.h"
#include "mathUtil.h"
#include "modelFullMat.h"

class ModelFullHinge: public ModelFullMat {
  public:
    ModelFullHinge(const Params& params, int nFeatures):ModelFullMat(params, nFeatures){}
    virtual void train(const Data &data, Model& bestModel);

};

#endif
