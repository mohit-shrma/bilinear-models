#include "modelFactSym.h"


ModelFactSym::ModelFactSym(const Params &params, int nFeatures)
  :Model(params, nFeatures) {
    //initialized model matrix
  U = Eigen::MatrixXf(nFeatures, rank);
  for (int i = 0; i < nFeatures; i++) {
    for (int j = 0; j < rank; j++) {
        U(i, j) = (float)std::rand()/ (float)(1.0 + RAND_MAX);
    }
  }
  std::cout << "\nU dim: " << U.rows() << " " << U.cols();
}


float ModelFactSym::estPosRating(int u, int item, const Data& data,
      Eigen::VectorXf& pdt) {
  float r_ui; 
  Eigen::VectorXf pdt1(rank);
  Eigen::VectorXf pdt2(rank);
  pdt1.fill(0);
  pdt2.fill(0);
  
  //(f_u^TU - f_i^TU)
  spVecMatPdt(U, data.uFAccumMat, u, pdt1); 
  spVecMatPdt(U, data.itemFeatMat, item, pdt2);
  pdt1 = pdt1 - pdt2;

  //f_i^TU
  spVecMatPdt(U, data.itemFeatMat, item, pdt2);

  r_ui = pdt1.dot(pdt2);
  return r_ui;
}


float ModelFactSym::estNegRating(int u, int item, const Data& data,
      Eigen::VectorXf& pdt) {
  float r_ui; 
  Eigen::VectorXf pdt1(rank);
  Eigen::VectorXf pdt2(rank);
  pdt1.fill(0);
  pdt2.fill(0);
  
  //(f_u^TU)
  spVecMatPdt(U, data.uFAccumMat, u, pdt1); 
  pdt1 = pdt1 - pdt2;

  //f_i^TU
  spVecMatPdt(U, data.itemFeatMat, item, pdt2);

  r_ui = pdt1.dot(pdt2);
  return r_ui;

}

