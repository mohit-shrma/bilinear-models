#ifndef _MODEL_LINEAR_H_
#define _MODEL_LINEAR_H_

#include "model.h"

class ModelLinear:public Model {

  public:
    ModelLinear(const Params &params, int nFeatures);
  
    virtual float estPosRating(int u, int item, const Data& data,
      Eigen::VectorXf& pdt);
    virtual float estNegRating(int u, int item, const Data& data, 
      Eigen::VectorXf& pdt);

};

#endif
