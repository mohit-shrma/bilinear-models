#ifndef _MODEL_FACT_BPR_H_
#define _MODEL_FACT_BPR_H_

#include "modelFactMat.h"

class ModelFactBPR: public ModelFactMat {

  public:
    ModelFactBPR(const Params &params, int nFeatures):ModelFactMat(params, nFeatures){}
    virtual void train(const Data &data, Model& bestModel);

}


#endif

