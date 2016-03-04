#include "modelBPRFullGrad.h"



void ModelBPRFGrad::train(const Data &data, Model& bestModel) {

  std::cout << "\nModelBPRFGrad::train" << std::endl;

  int bestIter, u, pI, nI;
  Eigen::MatrixXf Wgrad(nFeatures, nFeatures);  
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf jFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  Eigen::VectorXf pdt(nFeatures);
  float bestRecall, prevRecall, r_ui;
  int trainNNZ = getNNZ(data.trainMat); 
 
  std::cout << "\ntrain nnz: " << trainNNZ << " trainSamples: " << trainNNZ*pcSamples << std::endl;
  //std::cout << "val recall: " << computeRecallPar(data.valMat, data, 10, data.valItems) << std::endl;

  //random engine
  std::mt19937 mt(seed);
  
  auto uiRatings = getUIRatings(data.trainMat);
  
  std::chrono::time_point<std::chrono::system_clock> start, end;
  double regMult = (1.0 - 2.0*learnRate*l2Reg);
  for (int iter = 0; iter < maxIter; iter++) {
    //shuffle the user item ratings
    std::shuffle(uiRatings.begin(), uiRatings.end(), mt);
    start = std::chrono::system_clock::now();
    Wgrad.fill(0);
    int subIter = 0;
    for (auto&& uiRating: uiRatings) {
      //get user, item and rating
      u    = std::get<0>(uiRating);
      pI   = std::get<1>(uiRating);
      r_ui = std::get<2>(uiRating);
      
      //skip if negative or 0 rating
      if (r_ui <= 0) {
        continue;
      }

      if (data.posTrainUsers.find(u) != data.posTrainUsers.end()) {
        //found u, not valid train user
        continue;
      }

      //sample a negative item for user u
      nI = data.sampleNegItem(u);

      float r_uj = estNegRating(u, nI, data, pdt);
      double r_uij = r_ui - r_uj;
      double expCoeff = 1.0 /(1.0 + exp(r_uij));
      
      //- expCoeff * f_u * f_i^T
      updateMatWSpOuterPdt(Wgrad, data.uFAccumMat, u, data.itemFeatMat, pI, 
          -expCoeff);
      
      // expCoeff * f_u * f_j^T
      updateMatWSpOuterPdt(Wgrad, data.uFAccumMat, u, data.itemFeatMat, nI, 
          expCoeff);

      // expCoeff * f_i * f_i^T
      updateMatWSpOuterPdt(Wgrad, data.itemFeatMat, pI, data.itemFeatMat, pI, 
          expCoeff);
      
      subIter++;
    } 
    
    //update W
    W = (W*regMult) - ((learnRate/subIter) *  Wgrad);
    
    //nuclear norm projection after all sub-iters
    std::chrono::time_point<std::chrono::system_clock> startSVD, endSVD;
    startSVD = std::chrono::system_clock::now();
    //performNucNormProjSVDLib(W, rank);
    //performNucNormProjSVDLibWReg(W, nucReg);
    
    endSVD = std::chrono::system_clock::now(); 
    std::chrono::duration<double> durationSVD =  (endSVD - startSVD) ;
    std::cout << "\nsvd duration: " << durationSVD.count();
 
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = end - start;
    
    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0 || iter == maxIter - 1) {
      if(isTerminateModel(bestModel, data, iter, bestIter, bestRecall, 
          prevRecall)) {
        break;
      }
      std::cout << "\niter: " << iter << " val recall: " << prevRecall
        << " best recall: " << bestRecall << " duration: " 
        << duration.count() << std::endl;
    }
  
  }
  
}



