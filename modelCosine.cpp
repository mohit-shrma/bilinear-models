#include "modelCosine.h"

float ModelCosine::estNegRating(int u, int item, const Data& data,
    Eigen::VectorXf& pdt) {

  float rating = sparseDotProd2(data.uFAccumMat, u, data.itemFeatMat, item);
  return rating;
}



