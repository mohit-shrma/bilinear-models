#include "modelCosine.h"

float ModelCosine::estNegRating(int u, int item, const Data& data,
    Eigen::VectorXf& pdt) {
  float rating = sparseDotProd(data.itemFeatMat, item, data.uFAccumMat, u);
  return rating;
}

float ModelCosine::estPosRating(int u, int item, const Data& data,
    Eigen::VectorXf& pdt) {
  float rating = sparseDotProd(data.itemFeatMat, item, data.uFAccumMat, u);
  return rating;
}
