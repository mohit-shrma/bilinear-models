#include "modelRMSE.h"


float ModelRMSE::objective(const Data& data) {
  
  int u, ii, item, nnz;
  float r_ui, r_ui_est, w_norm;
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  Eigen::VectorXf pdt(nFeatures);
  float rmse = 0, WReg = 0, nucNormReg = 0, nucNorm = 0, obj = 0;
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
  WReg = l2Reg*w_norm*w_norm;
 
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

  obj += rmse/nnz + WReg + nucNormReg;
  
  return obj;
}


void ModelRMSE::computeRMSEGrad(Eigen::VectorXf& uFeat, Eigen::VectorXf& iFeat, Eigen::MatrixXf& Wgrad, float r_ui) {
  float r_ui_est;
  
  r_ui_est  = (uFeat - iFeat).transpose()*W*iFeat;
  
  //std::cout << "\nr_ui: " << r_ui << " r_ui_est: " << r_ui_est << std::endl;

  //reset Wgrad to gradient of l2 reg
  Wgrad = 2.0*l2Reg*W;
  Wgrad += 2.0*(r_ui_est - r_ui)*((uFeat - iFeat)*iFeat.transpose());
}


void ModelRMSE::computeRMSESparseGrad(int u, int i, float r_ui, 
    Eigen::MatrixXf& Wgrad, Eigen::VectorXf& pdt, const Data& data) {
  
  float r_ui_est;
  float l2RegTwice = 2.0*l2Reg;

  r_ui_est  = estPosRating(u, i, data, pdt);
  
  Wgrad = l2RegTwice*W;

  float outWt = 2.0*(r_ui_est - r_ui); 

  //f_u*f_i^T
  updateMatWSpOuterPdt(Wgrad, data.uFAccumMat, u, data.itemFeatMat, i, outWt);
  
  //-f_i*f_i^T
  updateMatWSpOuterPdt(Wgrad, data.itemFeatMat, i, data.itemFeatMat, i, -1*outWt);
}


//gradient check
void ModelRMSE::gradCheck(Eigen::VectorXf& uFeat, Eigen::VectorXf& iFeat, 
    Eigen::MatrixXf& Wgrad, float r_ui) {
  
  float r_ui_est, lossRight, lossLeft, gradE, w_norm;
  
  r_ui_est  = (uFeat - iFeat).transpose()*W*iFeat;
   
  //reset Wgrad to gradient of l2 reg
  Wgrad = 2.0*l2Reg*W;
  Wgrad += 2.0*(r_ui_est - r_ui)*((uFeat - iFeat)*iFeat.transpose());

  //perturbation matrix
  auto perturbMat = Eigen::MatrixXf::Constant(W.rows(), W.cols(), 0.0001);

  //perturb W with +E and compute loss
  auto noisyW1 = W + perturbMat; 
  r_ui_est  = (uFeat - iFeat).transpose()*noisyW1*iFeat;
  w_norm = noisyW1.norm();
  lossRight = (r_ui_est - r_ui)*(r_ui_est - r_ui) + l2Reg*w_norm*w_norm;

  //perturb W with -E and compute loss
  auto noisyW2 = W - perturbMat; 
  r_ui_est  = (uFeat - iFeat).transpose()*noisyW2*iFeat;
  w_norm = noisyW2.norm();
  lossLeft = (r_ui_est - r_ui)*(r_ui_est - r_ui) + l2Reg*w_norm*w_norm;

  //compute gradient and E dotprod
  gradE = 2.0*(Wgrad.cwiseProduct(perturbMat).sum());
  
  if (fabs(lossRight - lossLeft - gradE) > 0.01) {
    std::cout << "\nlr: " << lossRight << " ll: " << lossLeft 
      << " gradE: " << gradE  
      << " lr-ll: " << lossRight-lossLeft 
      << "\n(lr-ll)/gradE: " << (lossRight-lossLeft)/gradE  
      << " lr-ll-gradE: " << lossRight-lossLeft-gradE << std::endl;
  }

}

/*
void ModelRMSE::train2(const Data &data, Model& bestModel) {
  
  std::cout << "\nModelRMSE::train";

  int bestIter;
  Eigen::MatrixXf Wgrad(nFeatures, nFeatures);  
  Eigen::MatrixXf::Zero T(nFeatures, nFeatures);
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  Eigen::VectorXf pdt(nFeatures);
  float bestObj, prevObj, valRecall;
  int trainNNZ = getNNZ(data.trainMat); 

  int u, ii, item, nUserItems;
  float r_ui;
   
  std::cout <<"\nB4 Train Objective: " << objective(data) << std::endl;

  std::chrono::time_point<std::chrono::system_clock> startSub, endSub;
  std::chrono::time_point<std::chrono::system_clock> startGrad, endGrad;
  std::chrono::time_point<std::chrono::system_clock> startRat, endRat;
  std::chrono::time_point<std::chrono::system_clock> startOut, endOut;
  std::chrono::time_point<std::chrono::system_clock> startUpdGrad, endUpdGrad;
  std::chrono::duration<double> durationGrad, durationRat;
  double gradDuration = 0;
  double ratDuration = 0;
  double outDuration, updDuration;
  for (int iter = 0; iter < maxIter; iter++) {
    startSub = std::chrono::system_clock::now();
    gradDuration = 0;
    ratDuration = 0;
    outDuration = 0;
    updDuration = 0;
    for (int subIter = 0; subIter < trainNNZ; subIter++) {
    //for (int subIter = 0; subIter < 10; subIter++) {
      
      //sample user
      while (1) {
        u = std::rand() % data.nUsers;
        if (data.posTrainUsers.find(u) != data.posTrainUsers.end()) {
          //found u
          break;
        }
      }
      
      //sample a rated item
      nUserItems = data.trainMat->rowptr[u+1] - data.trainMat->rowptr[u];
      ii = std::rand()%nUserItems + data.trainMat->rowptr[u];
      item = data.trainMat->rowind[ii];
      r_ui = data.trainMat->rowval[ii];
      
      //extractFeat(data.uFAccumMat, u, uFeat);
      //extractFeat(data.itemFeatMat, item, iFeat);
      
      startGrad = std::chrono::system_clock::now();
      //compute gradient
      //gradCheck(uFeat, iFeat, Wgrad, r_ui);
      //computeRMSEGrad(uFeat, iFeat, Wgrad, r_ui);
      
      computeRMSESparseGrad(u, item, r_ui, Wgrad, pdt, data, startRat, endRat,
          startOut, endOut, startUpdGrad, endUpdGrad);
      
      endGrad = std::chrono::system_clock::now();
      durationGrad = endGrad - startGrad;
      gradDuration += durationGrad.count();

      durationRat = endRat - startRat;
      ratDuration += durationRat.count();

      durationRat = endOut - startOut;
      outDuration += durationRat.count();

      durationRat = endUpdGrad - startUpdGrad;
      updDuration += durationRat.count();

      //update W
      W -= learnRate*Wgrad;      
      

        //std::cout << "\n" << W.block<3,3>(0,0) << std::endl;
      //std::cout << W << std::endl;
    }
    endSub = std::chrono::system_clock::now(); 
    std::chrono::duration<double> durationSub =  (endSub - startSub) ;
    std::cout << "\nsubiter duration: " << durationSub.count();
    std::cout << "\ngradient duration: " << gradDuration;
    std::cout << "\nrating est duration: " << ratDuration; 
    std::cout << "\nout duration: " << outDuration;
    std::cout << "\nupd duration: " << updDuration << std::endl;

    //nuclear norm projection after all sub-iters
    std::chrono::time_point<std::chrono::system_clock> startSVD, endSVD;
    startSVD = std::chrono::system_clock::now();
    
    //performNucNormProjSVDLib(W, rank);
    performNucNormProjSVDLibWReg(W, nucReg);

    endSVD = std::chrono::system_clock::now(); 
    std::chrono::duration<double> durationSVD =  (endSVD - startSVD) ;
    std::cout << "\nsvd duration: " << durationSVD.count();
    
    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0) {
      //valRecall = computeRecallPar(data.valMat, data, 10, data.valItems);
      if (isTerminateModelObj(bestModel, data, iter, bestIter, bestObj, 
          prevObj)) {
        break;
      }
      std::cout << "\nIter: " << iter << " obj: " << prevObj
        << " best iter: " << bestIter << " best obj: " << bestObj << std::endl;
      std::cout << "\nTrain RMSE: " << computeRMSE(data.trainMat, data) << std::endl;
    }

  }

}
*/

void ModelRMSE::train(const Data &data, Model& bestModel) {
  
  std::cout << "\nModelRMSE::train";

  int bestIter;
  Eigen::MatrixXf Wgrad(nFeatures, nFeatures);  
  Eigen::MatrixXf T(nFeatures, nFeatures);
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  Eigen::VectorXf pdt(nFeatures);
  float bestObj, prevObj, valRecall;
  int trainNNZ = getNNZ(data.trainMat); 

  int u, ii, item, nUserItems;
  float r_ui;
  float r_ui_est; 
  std::cout <<"\nB4 Train Objective: " << objective(data) << std::endl;

  std::chrono::time_point<std::chrono::system_clock> startSub, endSub;
  double regMult =  (1.0 - 2.0*learnRate*l2Reg);
  for (int iter = 0; iter < maxIter; iter++) {
    startSub = std::chrono::system_clock::now();
    T.fill(0);
    int subIter;
    for (subIter = 0; subIter < trainNNZ; subIter++) {
      
      //sample user
      while (1) {
        u = std::rand() % data.nUsers;
        if (data.posTrainUsers.find(u) != data.posTrainUsers.end()) {
          //found u
          break;
        }
      }
      
      //sample a rated item
      nUserItems = data.trainMat->rowptr[u+1] - data.trainMat->rowptr[u];
      ii = std::rand()%nUserItems + data.trainMat->rowptr[u];
      item = data.trainMat->rowind[ii];
      r_ui = data.trainMat->rowval[ii];
      r_ui_est = estPosRating(u, item, data, pdt);
        
      double err = 2.0*(r_ui_est - r_ui);

      //- learnRate * err * f_u * f_i^T
      lazyUpdMatWSpOuterPdt(W, T, data.uFAccumMat, u, data.itemFeatMat, item,
        -learnRate*err, regMult, subIter);

      //learnRate * err * f_i * f_i^T
      lazyUpdMatWSpOuterPdt(W, T, data.itemFeatMat, item, data.itemFeatMat, item,
          learnRate*err, regMult, subIter);
    }

    //perform reg updates on all the pairs
    for (int ind1 = 0; ind1 < nFeatures; ind1++) {
      for (int ind2 = 0; ind2 < nFeatures; ind2++) {
        //update with reg updates
         W(ind1, ind2) = W(ind1, ind2)*pow(regMult, 
                                           subIter-T(ind1, ind2));
      }
    }
    
    endSub = std::chrono::system_clock::now(); 
    std::chrono::duration<double> durationSub =  (endSub - startSub) ;
    std::cout << "\nsubiter duration: " << durationSub.count() << std::endl;

    //nuclear norm projection after all sub-iters
    std::chrono::time_point<std::chrono::system_clock> startSVD, endSVD;
    startSVD = std::chrono::system_clock::now();
    
    //performNucNormProjSVDLib(W, rank);
    //performNucNormProjSVDLibWReg(W, nucReg);

    endSVD = std::chrono::system_clock::now(); 
    std::chrono::duration<double> durationSVD =  (endSVD - startSVD) ;
    std::cout << "\nsvd duration: " << durationSVD.count();
    
    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0) {
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
