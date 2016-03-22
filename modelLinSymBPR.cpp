#include "modelLinSymBPR.h"


void ModelLinSymBPR::computeUGrad(int u, int i, int j, const Data& data, 
    Eigen::MatrixXf& Ugrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat,
    Eigen::VectorXf& uFeat) { 
  
  float r_ui = ((uFeat - iFeat).transpose()*U)*(U.transpose()*iFeat);
  r_ui  += (((uFeat - iFeat).transpose())*(iFeat.cwiseProduct(w)));
  float r_uj = (uFeat.transpose()*U)*(U.transpose()*jFeat);
  r_uj  += uFeat.transpose()*(jFeat.cwiseProduct(w));
  float r_uij = r_ui - r_uj;
  float expCoeff = -1.0/(1.0 + exp(r_uij));
  Ugrad = (uFeat - iFeat)*(iFeat.transpose()*U) + iFeat*((uFeat - iFeat).transpose()*U);
  Ugrad -= (uFeat*(jFeat.transpose()*U) + jFeat*(uFeat.transpose()*U));
  Ugrad *= expCoeff;
  //reg
  Ugrad += 2.0*l2Reg*U;
} 


void ModelLinSymBPR::computewGrad(int u, int i, int j, const Data& data, 
    Eigen::VectorXf& wgrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat, 
    Eigen::VectorXf& uFeat) {
  float r_ui = ((uFeat - iFeat).transpose()*U)*(U.transpose()*iFeat);
  r_ui  += (((uFeat - iFeat).transpose())*(iFeat.cwiseProduct(w)));
  float r_uj = (uFeat.transpose()*U)*(U.transpose()*jFeat);
  r_uj  += uFeat.transpose()*(jFeat.cwiseProduct(w));
  float r_uij = r_ui - r_uj;
  float expCoeff = -1.0/(1.0 + exp(r_uij));
  wgrad = ((uFeat - iFeat).cwiseProduct(iFeat)) - uFeat.cwiseProduct(jFeat);
  wgrad *= expCoeff;
  wgrad += 2.0*wl2Reg*w;
}



void ModelLinSymBPR::train(const Data &data, Model& bestModel) {

  std::cout << "\nModelLinSymBPR::train" << std::endl;

  int bestIter, u, i, j;
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf jFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  Eigen::VectorXf wgrad(nFeatures);
  Eigen::MatrixXf Ugrad(nFeatures, rank);  

  float bestRecall, prevRecall;
  int trainNNZ = getNNZ(data.trainMat); 
 
  std::cout << "\ntrain nnz: " << trainNNZ << " trainSamples: " << trainNNZ*pcSamples << std::endl;
  std::cout << "val recall: " << computeRecallPar(data.valMat, data, 10, data.valItems) << std::endl;
  //random engine
  std::mt19937 mt(seed);
  
  auto uiRatings = getBPRUIRatings(data.trainMat);
  std::cout << "\nuiRatings: " << uiRatings.size();
  
  for (int iter = 0; iter < maxIter; iter++) {
    //shuffle the user item ratings
    std::shuffle(uiRatings.begin(), uiRatings.end(), mt);
    for (auto&& uiRating: uiRatings) {
      //get user, item and rating
      u = std::get<0>(uiRating);
      i = std::get<1>(uiRating);

      //sample a negative item for user u
      j = data.sampleNegItem(u);
      
      extractFeat(data.uFAccumMat, u, uFeat);
      extractFeat(data.itemFeatMat, i, iFeat);
      extractFeat(data.itemFeatMat, j, jFeat);

      //compute w gradient
      computewGrad(u, i, j, data, wgrad, iFeat, jFeat, uFeat);
      w -= learnRate*wgrad;
      
      //compute U gradient
      computeUGrad(u, i, j, data, Ugrad, iFeat, jFeat, uFeat);
      U -= learnRate*Ugrad;

    } 
    
    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0 || iter == maxIter - 1) {
      if(isTerminateModel(bestModel, data, iter, bestIter, bestRecall, 
          prevRecall)) {
        break;
      }
      std::cout << std::endl <<"iter: " << iter << " val recall: " << prevRecall
        << " best recall: " << bestRecall << " w norm: " << w.norm()
        << " U norm: " << U.norm() ;
    }
  
  }
  
}


