#include "modelRMSEFullGrad.h"


float ModelRMSEFGrad::objective(const Data& data) {
  
  int u, ii, item, nnz;
  float r_ui, r_ui_est, w_norm;
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  Eigen::VectorXf pdt(nFeatures);
  float rmse = 0, WL2Reg = 0, nucNormReg = 0, nucNorm = 0, obj = 0;
  float WL1Reg = 0;
  nnz = 0;
  for (u = 0; u < data.trainMat->nrows; u++) {
    extractFeat(data.uFAccumMat, u, uFeat);
    for (ii = data.trainMat->rowptr[u]; 
        ii < data.trainMat->rowptr[u+1]; ii++) {
      item = data.trainMat->rowind[ii];
      //extractFeat(data.itemFeatMat, item, iFeat);
      //r_ui_est = (uFeat - iFeat).transpose()*W*iFeat;
      r_ui_est = estPosRating(u, item, data, pdt);
      r_ui = data.trainMat->rowval[ii];
      rmse += (r_ui_est - r_ui)*(r_ui_est-r_ui);
      nnz++;
    }
  }
    
  w_norm = W.norm();
  WL2Reg = l2Reg*w_norm*w_norm;
 
  //WL1Reg = l1Reg*(W.lpNorm<1>());

  /*
  //compute thin svd
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(W, 
      Eigen::ComputeThinU|Eigen::ComputeThinV);
  auto singVec = svd.singularValues();
  for (ii = 0; ii < nFeatures; ii++) {
    nucNorm += singVec[ii]; 
  }
  nucNormReg = nucNorm*nucReg;
  */

  obj += rmse/nnz + WL2Reg + WL1Reg  + nucNormReg;
  
  return obj;
}



void ModelRMSEFGrad::train(const Data &data, Model& bestModel) {
  
  std::cout << "\nModelRMSEFGrad::train";

  int bestIter, subIter;
  Eigen::MatrixXf Wgrad(nFeatures, nFeatures);  
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  Eigen::VectorXf pdt(nFeatures);
  float bestObj, prevObj, valRecall;

  int u, item;
  float r_ui;
  float r_ui_est; 
  std::cout <<"\nB4 Train Objective: " << objective(data) << std::endl;

  //random engine
  std::mt19937 mt(seed);
  
  auto uiRatings = getUIRatings(data.trainMat);
  std::cout << "\nnRatings: " << uiRatings.size() << std::endl;
  std::chrono::time_point<std::chrono::system_clock> startSub, endSub;
  double regMult =  (1.0 - 2.0*learnRate*l2Reg);
  for (int iter = 0; iter < maxIter; iter++) {
    startSub = std::chrono::system_clock::now();
    Wgrad.fill(0);
    subIter = 0;
    for (auto&& uiRating: uiRatings) {
      //get user, item and rating
      u       = std::get<0>(uiRating);
      item    = std::get<1>(uiRating);
      r_ui    = std::get<2>(uiRating);
      
      if (data.posTrainUsers.find(u) != data.posTrainUsers.end()) {
        //found u
        continue;
      }
      
      r_ui_est = estPosRating(u, item, data, pdt);
        
      double err = 2.0*(r_ui_est - r_ui);
      
      //compute gradient

      //f_u*f_i^T
      updateMatWSpOuterPdt(Wgrad, data.uFAccumMat, u, data.itemFeatMat, item, err);
      
      //-f_i*f_i^T
      updateMatWSpOuterPdt(Wgrad, data.itemFeatMat, item, data.itemFeatMat, item, -1*err);

      subIter++;
    }

    //update W
    W = (W*regMult) - ((learnRate/subIter) *  Wgrad);
    
    //nuclear norm projection after all sub-iters
    std::chrono::time_point<std::chrono::system_clock> startSVD, endSVD;
    startSVD = std::chrono::system_clock::now();
    //performNucNormProjSVDLib(W, rank);
    performNucNormProjSVDLibWReg(W, nucReg);
    
    endSVD = std::chrono::system_clock::now(); 
    std::chrono::duration<double> durationSVD =  (endSVD - startSVD) ;
    std::cout << "\nsvd duration: " << durationSVD.count();
    
    endSub = std::chrono::system_clock::now(); 
    std::chrono::duration<double> durationSub =  (endSub - startSub) ;
    std::cout << "\nsubiter duration: " << durationSub.count() << std::endl;
    
    //TODO: objective

    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0 || iter == maxIter - 1) {
      //valRecall = computeRecallPar(data.valMat, data, 10, data.valItems);
      if (isTerminateModelObj(bestModel, data, iter, bestIter, bestObj, 
          prevObj)) {
        break;
      }
      std::cout << "\nIter: " << iter << " obj: " << prevObj
        << " best iter: " << bestIter << " best obj: " << bestObj;
      std::cout << "\nTrain RMSE: " << computeRMSE(data.trainMat, data) << std::endl;
    }

  }

}



