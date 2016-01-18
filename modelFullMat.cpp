#include "modelFullMat.h"

ModelFullMat::ModelFullMat(const Params &params, int nFeatures):Model(params, nFeatures){
  //initialize model matrix
  W = Eigen::MatrixXf::Zero(nFeatures, nFeatures);
  for (int i = 0; i < nFeatures; i++) {
    for (int j = 0; j < nFeatures; j++) {
      if (i == j) {
        //initialize diagonal as 1 since cosine sim is good satrting point
        W(i, j) = 1.0;
      } else {
        W(i, j) = (float)std::rand()/ (float)(1.0 + RAND_MAX);
      }
    }
  }
  std::cout << "\nW norm: " << W.norm() << std::endl;
}


//estimate rating on a positively rated item
float ModelFullMat::estPosRating(int u, int item, const Data& data,
    Eigen::VectorXf& pdt) {
  float r_ui = 0;
  pdt.fill(0);

  //compute dot product of mat and sparse vector 
  matSpVecPdt(W, data.itemFeatMat, item, pdt);
  //f_u^TWf_i 
  r_ui = vecSpVecDot(pdt, data.uFAccumMat, u);
  //-f_i^TWf_i
  r_ui -= vecSpVecDot(pdt, data.itemFeatMat, item);

  return r_ui; 
} 


//estimate rating on a item not rated before
float ModelFullMat::estNegRating(int u, int item, const Data& data, 
    Eigen::VectorXf& pdt) {
  float r_ui = 0;
  pdt.fill(0);

  //compute dot product of mat and sparse vector 
  matSpVecPdt(W, data.itemFeatMat, item, pdt);
  r_ui = vecSpVecDot(pdt, data.uFAccumMat, u);

  return r_ui;
}





