#ifndef _MODEL_FULL_MAT_H_
#define _MODEL_FULL_MAT_H_


#include "model.h"

class ModelFullMat: public Model {
  
  public:
    Eigen::MatrixXf W;
    ModelFullMat(const Params &params, int nFeatures);
    
    virtual float estPosRating(int u, int item, const Data& data,
      Eigen::VectorXf& pdt);
    virtual float estNegRating(int u, int item, const Data& data, 
      Eigen::VectorXf& pdt);


};

#endif
