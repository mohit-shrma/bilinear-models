#include "modelFactBPR.h"

void computeUGrad(int u, int i, int j, const Data& data, 
    Eigen::MatrixXf& Ugrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat,
    Eigen::VectorXf& uFeat) {

  float r_uij = ((uFeat - iFeat).transpose()*U)*(V.transpose()*iFeat) -
                (uFeat.transpose()*U)*(V.transpose()*jFeat);
  float expCoeff = 1.0/(1.0 + r_uij);
  Ugrad = (uFeat - iFeat)*(iFeat.transpose()*V) - uFeat*(jFeat.transpose()*V);
  Ugrad *= expCoeff;
  //TODO: regularization
}


void computeVGrad(int u, int i, int j, const Data& data, 
    Eigen::MatrixXf& Vgrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat,
    Eigen::VectorXf& uFeat) {

  float r_uij = ((uFeat - iFeat).transpose()*U)*(V.transpose()*iFeat) -
                (uFeat.transpose()*U)*(V.transpose()*jFeat);
  float expCoeff = 1.0/(1.0 + r_uij);
  Vgrad = (iFeat - jFeat)*(uFeat.transpose()*U) - iFeat*(iFeat.transpose()*U);
  Vgrad *= expCoeff;
  //TODO: regularization
}


void ModelFactBPR::train(const Data &data, Model& bestModel) {

  std::cout << "\nModelFactBPR::train" << std::endl;

  int bestIter, u, i, j;
  Eigen::MatrixXf Ugrad(nFeatures, rank);  
  Eigen::MatrixXf Vgrad(nFeatures, rank);  
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf jFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  float bestRecall, prevRecall;
  int trainNNZ = getNNZ(data.trainMat); 
  std::array<int, 3> triplet;
 
  std::cout << "\ntrain nnz: " << trainNNZ << " trainSamples: " << trainNNZ*pcSamples << std::endl;

  for (int iter = 0; iter < maxIter; iter++) {
    for (int subIter = 0; subIter < trainNNZ*pcSamples; subIter++) {
        
      //sample triplet
      triplet = data.sampleTriplet();
      u = triplet[0];
      i = triplet[1];
      j = triplet[2];
      uFeat = data.uFeatAcuum.row(u); 
      extractFeat(data.itemFeatMat, i, iFeat);
      extractFeat(data.itemFeatMat, j, jFeat);
      
      //compute U gradient
      computeUGrad(u, i, j, data, Ugrad, iFeat, jFeat, uFeat);
      //update U
      U -= learnRate*Ugrad;

      //compute V gradient
      computeVGrad(u, i, j, data, Vgrad, iFeat, jFeat, uFeat);
      //update V
      V -= learnRate*Vgrad;
    } 
    
    
    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0) {
      if(isTerminateModel(bestModel, data, iter, bestIter, bestRecall, 
          prevRecall)) {
        break;
      }
      std::cout << "\niter: " << iter << " val recall: " << prevRecall
        << " best recall: " << bestRecall << std::endl;
    }
  
  }
  
}

