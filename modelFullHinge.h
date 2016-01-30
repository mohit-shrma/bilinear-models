#ifndef _MODEL_FULL_HINGE_H
#define _MODEL_FULL_HINGE_H

#include "model.h"
#include "mathUtil.h"
#include "modelFullMat.h"
#include <map>

class ModelFullHinge: public ModelFullMat {
  public:
    ModelFullHinge(const Params& params, int nFeatures):ModelFullMat(params, nFeatures){}
    virtual void train(const Data &data, Model& bestModel);
    int estRatingsforUser(int u, const Data& data, std::map<int, float>& itemRatings);
    void computeGrad(int u, Eigen::MatrixXf& Wgrad, 
      Eigen::MatrixXf& gradNegHull, const Data& data, int maxNegItem, 
      std::map<int, float>& itemRatings);
};

#endif
