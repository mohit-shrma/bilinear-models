#ifndef _COSINE_MODEL_H_
#define _COSINE_MODEL_H_


#include "model.h"

class ModelCosine: public Model {

  public:
    ModelCosine(const Params &params, int nFeatures):Model(params, nFeatures){}
    virtual float estNegRating(int u, int item, const Data& data, 
      Eigen::VectorXf& pdt);
};



#endif

