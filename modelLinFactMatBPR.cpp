#include "modelLinFactMatBPR.h"


void ModelLinFactMatBPR::computeUGrad(int u, int i, int j, const Data& data, 
    Eigen::MatrixXf& Ugrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat,
    Eigen::VectorXf& uFeat) {
  float r_ui = ((uFeat - iFeat).transpose()*U)*(V.transpose()*iFeat); 
  r_ui  += (((uFeat - iFeat).transpose())*(iFeat.cwiseProduct(w)));
  float r_uj = (uFeat.transpose()*U)*(V.transpose()*jFeat);
  r_uj  += uFeat.transpose()*(jFeat.cwiseProduct(w));
  float r_uij = r_ui - r_uj;
  float expCoeff = -1.0/(1.0 + exp(r_uij));
  Ugrad = (uFeat - iFeat)*(iFeat.transpose()*V) - uFeat*(jFeat.transpose()*V);
  Ugrad *= expCoeff;
  //regularization
  Ugrad += 2.0*l2Reg*U;
}


void ModelLinFactMatBPR::computeVGrad(int u, int i, int j, const Data& data, 
    Eigen::MatrixXf& Vgrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat,
    Eigen::VectorXf& uFeat) {
  float r_ui = ((uFeat - iFeat).transpose()*U)*(V.transpose()*iFeat); 
  r_ui  += (((uFeat - iFeat).transpose())*(iFeat.cwiseProduct(w)));
  float r_uj = (uFeat.transpose()*U)*(V.transpose()*jFeat);
  r_uj  += uFeat.transpose()*(jFeat.cwiseProduct(w));
  float r_uij = r_ui - r_uj;
  float expCoeff = -1.0/(1.0 + exp(r_uij));
  Vgrad = (iFeat - jFeat)*(uFeat.transpose()*U) - iFeat*(iFeat.transpose()*U);
  Vgrad *= expCoeff;
  //regularization
  Vgrad += 2.0*l2Reg*V;
}


void ModelLinFactMatBPR::computewGrad(int u, int i, int j, const Data& data, 
    Eigen::VectorXf& wgrad, Eigen::VectorXf& iFeat, Eigen::VectorXf& jFeat, 
    Eigen::VectorXf& uFeat) {
  float r_ui = ((uFeat - iFeat).transpose()*U)*(V.transpose()*iFeat); 
  r_ui  += (((uFeat - iFeat).transpose())*(iFeat.cwiseProduct(w)));
  float r_uj = (uFeat.transpose()*U)*(V.transpose()*jFeat);
  r_uj  += uFeat.transpose()*(jFeat.cwiseProduct(w));
  float r_uij = r_ui - r_uj;
  float expCoeff = -1.0/(1.0 + exp(r_uij));
  wgrad = ((uFeat - iFeat).cwiseProduct(iFeat)) - uFeat.cwiseProduct(jFeat);
  wgrad *= expCoeff;
  wgrad += 2.0*wl2Reg*w;
}


void ModelLinFactMatBPR::train(const Data &data, Model& bestModel) {

  std::cout << "\nModelLinFactMatBPR::train" << std::endl;

  int bestIter, u, i, j;
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf jFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  Eigen::VectorXf wgrad(nFeatures);
  Eigen::MatrixXf Ugrad(nFeatures, rank);  
  Eigen::MatrixXf Vgrad(nFeatures, rank);  

  float bestRecall, prevRecall;
  int trainNNZ = getNNZ(data.trainMat); 
 
  std::cout << "\ntrain nnz: " << trainNNZ << " trainSamples: " << trainNNZ*pcSamples << std::endl;
  //std::cout << "val recall: " << computeRecallPar(data.valMat, data, 10, data.valItems) << std::endl;
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

      //compute V gradient
      computeVGrad(u, i, j, data, Vgrad, iFeat, jFeat, uFeat);
      V -= learnRate*Vgrad;
    } 
    
    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0 || iter == maxIter - 1) {
      if(isTerminatewUVTModel(bestModel, data, iter, bestIter, bestRecall, 
          prevRecall)) {
        break;
      }
      std::cout << std::endl <<"iter: " << iter << " val recall: " << prevRecall
        << " best recall: " << bestRecall << " w norm: " << w.norm()
        << " U norm: " << U.norm() << " V norm: " << V.norm();
    }
  
  }
  
}


void ModelLinFactMatBPR::updateTriplets(
    std::vector<std::tuple<int, int, int>>& triplets, const Data& data,
    int start, int end) {
  
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf jFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  Eigen::VectorXf wgrad(nFeatures);
  Eigen::MatrixXf Ugrad(nFeatures, rank);  
  Eigen::MatrixXf Vgrad(nFeatures, rank);  
  
  wgrad.fill(0);
  Ugrad.fill(0);
  Vgrad.fill(0);

  for (int k = start; k < end; k++) {
    int u = std::get<0>(triplets[k]);
    int i = std::get<1>(triplets[k]);
    int j = std::get<2>(triplets[k]);
    extractFeat(data.uFAccumMat, u, uFeat);
    extractFeat(data.itemFeatMat, i, iFeat);
    extractFeat(data.itemFeatMat, j, jFeat);
    //compute w gradient
    computewGrad(u, i, j, data, wgrad, iFeat, jFeat, uFeat);
    w -= learnRate*wgrad;

    //compute U gradient
    computeUGrad(u, i, j, data, Ugrad, iFeat, jFeat, uFeat);
    U -= learnRate*Ugrad;

    //compute V gradient
    computeVGrad(u, i, j, data, Vgrad, iFeat, jFeat, uFeat);
    V -= learnRate*Vgrad;
  }

}


void ModelLinFactMatBPR::parTrain(const Data &data, Model& bestModel) { 

  std::cout << "\nModelLinFactMatBPR::train" << std::endl;

  int bestIter, u, i, j;
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf jFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  Eigen::VectorXf wgrad(nFeatures);
  Eigen::MatrixXf Ugrad(nFeatures, rank);  
  Eigen::MatrixXf Vgrad(nFeatures, rank);  

  float bestRecall, prevRecall;
  int trainNNZ = getNNZ(data.trainMat); 
 
  std::cout << "\ntrain nnz: " << trainNNZ << " trainSamples: " << trainNNZ*pcSamples << std::endl;
  //std::cout << "val recall: " << computeRecallPar(data.valMat, data, 10, data.valItems) << std::endl;
  //random engine
  std::mt19937 mt(seed);
  
  unsigned long const hwThreads = std::thread::hardware_concurrency();
  int nThreads = NTHREADS;
  if (hwThreads > 0  && hwThreads < NTHREADS) {
    nThreads = hwThreads;
  }

  std::cout << "nthreads: " << nThreads << std::endl;
  auto uiRatings = getBPRUIRatings(data.trainMat);
  std::cout << "\nuiRatings: " << uiRatings.size();
  
  std::vector<std::tuple<int, int, int>> bprTriplets;
  
  for (int iter = 0; iter < maxIter; iter++) { 
    //shuffle the user item ratings
    std::shuffle(uiRatings.begin(), uiRatings.end(), mt);
    bprTriplets.clear();
    for (auto&& uiRating: uiRatings) { 
      //get user, item and rating
      u = std::get<0>(uiRating);
      i = std::get<1>(uiRating);

      //sample a negative item for user u
      j = data.sampleNegItem(u);
      
      bprTriplets.push_back(std::make_tuple(u, i, j));
    } 
    
    std::cout << "bprTriplets: " << bprTriplets.size() << std::endl;
    
    //allocate thread
    std::vector<std::thread> threads(nThreads-1);
    std::vector<bool> threadDone(nThreads-1);
    std::fill(threadDone.begin(), threadDone.end(), false);
    
    int nRatingsPerThread = bprTriplets.size()/nThreads;
    std::cout << "nRatingsPerThread: " << nRatingsPerThread << std::endl;
    for (int thInd = 0; thInd < nThreads-1; thInd++) {
      int startInd = thInd*nRatingsPerThread;
      int endInd = (thInd+1)*nRatingsPerThread;
      threads[thInd] = std::thread(&ModelLinFactMatBPR::updateTriplets, this, std::ref(bprTriplets),
          std::ref(data), startInd, endInd);   
    } 


    //main thread
    updateTriplets(bprTriplets, data, (nThreads-1)*nRatingsPerThread, 
        bprTriplets.size());

    //wait for threads to join 
    std::for_each(threads.begin(), threads.end(), 
        std::mem_fn(&std::thread::join));
    
    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0 || iter == maxIter - 1) {
      if(isTerminatewUVTModel(bestModel, data, iter, bestIter, bestRecall, 
          prevRecall)) {
        break;
      }
      std::cout << std::endl <<"iter: " << iter << " val recall: " << prevRecall
        << " best recall: " << bestRecall << " w norm: " << w.norm()
        << " U norm: " << U.norm() << " V norm: " << V.norm();
    }
  
  }
  
}

