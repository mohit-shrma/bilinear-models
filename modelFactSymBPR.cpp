#include "modelFactSymBPR.h"


void ModelFactSymBPR::computeUGrad(int u, int i, int j, const Data& data, 
    Eigen::MatrixXf& Ugrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat,
    Eigen::VectorXf& uFeat, Eigen::MatrixXf& temp_nf_nf) {
  
  float r_ui = ((uFeat - iFeat).transpose()*U)*(U.transpose()*iFeat);
  float r_uj = (uFeat.transpose()*U)*(U.transpose()*jFeat);
  float r_uij = r_ui - r_uj;
  float expCoeff = -1.0/(1.0 + exp(r_uij));
  temp_nf_nf = (uFeat - iFeat)*iFeat.transpose() + iFeat*((uFeat - iFeat).transpose());
  temp_nf_nf -= (uFeat*jFeat.transpose() + jFeat*uFeat.transpose());
  Ugrad = temp_nf_nf*U;
  Ugrad *= expCoeff;
  //reg
  Ugrad += 2.0*l2Reg*U;
} 


void ModelFactSymBPR::computeUSpGrad(int u, int i, int j, const Data& data, 
    Eigen::MatrixXf& Ugrad, float r_uij_est, Eigen::VectorXf& f_iT_U,
    Eigen::VectorXf& f_uT_U, Eigen::VectorXf& f_jT_U, 
    Eigen::VectorXf& f_iT_U_f_jT_U, Eigen::VectorXf& f_uT_U_2f_iT_U) {
  
  float expCoeff = -1.0/(1.0 + exp(r_uij_est));
  Ugrad.fill(0);
  //-f_j(f_u^T*U)
  spVecVecOuterPdt(Ugrad, f_uT_U, data.itemFeatMat, j);
  Ugrad *= -1;
  //f_u(f_i^TU - f_j^TU)
  spVecVecOuterPdt(Ugrad, f_iT_U_f_jT_U, data.uFAccumMat, u);
  //f_i(f_u^T*U - 2*f_i^T*U)
  spVecVecOuterPdt(Ugrad, f_uT_U_2f_iT_U, data.itemFeatMat, i);
  Ugrad *= expCoeff;
  //reg
  Ugrad += 2.0*l2Reg*U;
}


//TODO: verify if model correctly copied in bestModel
void ModelFactSymBPR::train(const Data& data, Model& bestModel) {

  std::cout << "\nModelFactSymBPR::train";
  int bestIter, u, i, j;
  Eigen::MatrixXf temp(nFeatures, nFeatures);
  Eigen::MatrixXf Ugrad(nFeatures, rank);  
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf jFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  Eigen::VectorXf f_uT_U(rank);
  Eigen::VectorXf f_iT_U(rank);
  Eigen::VectorXf f_jT_U(rank);
  Eigen::VectorXf f_iT_U_f_jT_U(rank);
  Eigen::VectorXf f_uT_U_2f_iT_U(rank);

  float bestRecall, prevRecall, r_uij_est;
  int trainNNZ = getNNZ(data.trainMat); 
  std::array<int, 3> triplet;
 
  std::cout << "\ntrain nnz: " << trainNNZ << " trainSamples: " << trainNNZ*pcSamples << std::endl;
  std::cout << "val recall: " << computeRecallPar(data.valMat, data, 10, data.valItems) << std::endl;
  for (int iter = 0; iter < maxIter; iter++) {
    for (int subIter = 0; subIter < trainNNZ*pcSamples; subIter++) {
    //for (int subIter = 0; subIter < 10; subIter++) {
        
      //sample triplet
      triplet = data.sampleTriplet();
      u = triplet[0];
      i = triplet[1];
      j = triplet[2];
      
      uFeat = data.uFeatAcuum.row(u); 
      extractFeat(data.itemFeatMat, i, iFeat);
      extractFeat(data.itemFeatMat, j, jFeat);

      //gradCheck(u, i, j, Ugrad, Vgrad, data, uFeat, iFeat, jFeat);

      //compute f_u^TU
      spVecMatPdt(U, data.uFAccumMat, u, f_uT_U);
      
      //compute f_i^TU
      spVecMatPdt(U, data.itemFeatMat, i, f_iT_U);
      
      //compute f_j^TU
      spVecMatPdt(U, data.itemFeatMat, j, f_jT_U);
      
      f_iT_U_f_jT_U = f_iT_U - f_jT_U;
      f_uT_U_2f_iT_U = f_uT_U - 2*f_iT_U;

      //r_uij_est
      r_uij_est = f_uT_U.dot(f_iT_U - f_jT_U) - f_iT_U.dot(f_iT_U);

      //compute U gradient
      //computeUGrad(u, i, j, data, Ugrad, iFeat, jFeat, uFeat);
      computeUSpGrad(u, i, j, data, Ugrad, r_uij_est, f_iT_U, f_uT_U, f_jT_U,
          f_iT_U_f_jT_U, f_uT_U_2f_iT_U);
      //update U
      U -= learnRate*Ugrad;

    } 
    
    
    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0) {
      if(isTerminateModel(bestModel, data, iter, bestIter, bestRecall, 
          prevRecall)) {
        break;
      }
      std::cout << std::endl <<"iter: " << iter << " val recall: " << prevRecall
        << " best recall: " << bestRecall;
    }
  
  }
 

}



