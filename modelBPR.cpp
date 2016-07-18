#include "modelBPR.h"

bool ModelBPR::isTerminateModelInit(Model& bestModel, const Data& data, int iter,
    int& bestIter, float& bestRecall, float& prevRecall) {

  bool ret = false;
  float currRecall = computeRecallParVec(data.valMat, data, 10, data.valItems);
  
  if (iter >= 0) {
    
    if (currRecall > bestRecall) {
      bestModel = *this;
      bestRecall = currRecall;
      bestIter = iter;
    }
    
    if (iter - bestIter >= CHANCE_ITER) {
      std::cout << "\nNOT CONVERGED: bestIter: " << bestIter << 
        " bestRecall: " << bestRecall << " currIter: " << iter << 
        " currRecall: " << currRecall << std::endl;
      ret = true;
    }

    if (fabs(prevRecall - currRecall) < EPS) {
      //convergence
      std::cout << "\nConverged in iteration: " << iter << " prevRecall: "
        << prevRecall << " currRecall: " << currRecall;
      ret = true;
    }

  }


  prevRecall = currRecall;

  return ret;
}


void ModelBPR::computeBPRGrad(Eigen::VectorXf& uFeat, Eigen::VectorXf& iFeat, 
    Eigen::VectorXf& jFeat, Eigen::MatrixXf& Wgrad) {
  double r_ui, r_uj, r_uij, expCoeff;
  
  r_ui  = (uFeat - iFeat).transpose()*W*iFeat;
  r_uj  = uFeat.transpose()*W*jFeat;
  r_uij = r_ui - r_uj;
   
  Wgrad.fill(0);
  expCoeff = -1.0/(1.0 + exp(r_uij));  
  //need to update W as j has higher preference
  Wgrad = ((uFeat - iFeat)*iFeat.transpose()
                    - uFeat*jFeat.transpose());
  Wgrad *= expCoeff;
  Wgrad += 2.0*l2Reg*W;
}


void ModelBPR::computeBPRSparseGrad(int u, int i, int j, 
    Eigen::MatrixXf& Wgrad, Eigen::VectorXf& pdt, const Data& data,
    std::map<int, std::unordered_set<int>>& coords) {
  
  float r_ui, r_uj, r_uij, expCoeff;
   
  r_ui  = estPosRating(u, i, data, pdt);
  r_uj  = estNegRating(u, j, data, pdt);
  r_uij = r_ui - r_uj;
   
  Wgrad.setZero(nFeatures, nFeatures);
  coords.clear();

  //need to update W as j has higher preference
  expCoeff = 1.0/(1.0 + exp(r_uij));  

  //-f_u*f_i^T  * expCoeff
  updateMatWSpOuterPdtWMap(Wgrad, data.uFAccumMat, u, data.itemFeatMat, i, 
      -expCoeff, coords);

  //f_u*f_j^T  * expCoeff
  updateMatWSpOuterPdtWMap(Wgrad, data.uFAccumMat, u, data.itemFeatMat, j, 
      expCoeff, coords);

  //f_i*f_i^T  * expCoeff
  updateMatWSpOuterPdtWMap(Wgrad, data.itemFeatMat, i, data.itemFeatMat, i, 
      expCoeff, coords);
  
}


void ModelBPR::gradCheck(Eigen::VectorXf& uFeat, Eigen::VectorXf& iFeat, 
    Eigen::VectorXf& jFeat, Eigen::MatrixXf& Wgrad) {
  
  double r_ui, r_uj, r_uij, expCoeff;
  float lossRight, lossLeft, gradE;

  r_ui  = (uFeat - iFeat).transpose()*W*iFeat;
  r_uj  = uFeat.transpose()*W*jFeat;
  r_uij = r_ui - r_uj;

  if (r_uij > 0 ) {
    return;
  }

  expCoeff = 1.0/(1.0 + exp(r_uij));  
   
  //reset Wgrad to gradient of l2 reg
  //Wgrad = 2.0*l2Reg*W;
  Wgrad = -expCoeff*((uFeat - iFeat)*iFeat.transpose()
                      - uFeat*jFeat.transpose());
  //perturbation matrix
  auto perturbMat = Eigen::MatrixXf::Constant(W.rows(), W.cols(), 0.0001);

  //perturb W with +E and compute loss
  auto noisyW1 = W + perturbMat; 
  r_ui = (uFeat - iFeat).transpose()*noisyW1*iFeat;
  r_uj = uFeat.transpose()*noisyW1*jFeat;
  //w_norm = noisyW1.norm(); 
  lossRight = -log(sigmoid(r_ui - r_uj));// + l2Reg*w_norm*w_norm;

  //perturb W with -E and compute loss
  auto noisyW2 = W - perturbMat; 
  r_ui = (uFeat - iFeat).transpose()*noisyW2*iFeat;
  r_uj = uFeat.transpose()*noisyW2*jFeat;
  //w_norm = noisyW2.norm();
  lossLeft = -log(sigmoid(r_ui - r_uj));// + l2Reg*w_norm*w_norm;

  //compute gradient and E dotprod
  gradE = 2.0*(Wgrad.cwiseProduct(perturbMat).sum());

  if (fabs(lossRight - lossLeft - gradE) > 0.001) {
    std::cout << "\nlr: " << lossRight << " ll: " << lossLeft 
      << " gradE: " << gradE  
      << " lr-ll: " << lossRight-lossLeft 
      << "\n(lr-ll)/gradE: " << (lossRight-lossLeft)/gradE  
      << " lr-ll-gradE: " << lossRight-lossLeft-gradE
      << " uFeatNorm: " << uFeat.norm()
      << " iFeatNorm: " << iFeat.norm() << " jFeatNorm: " << jFeat.norm()
      << std::endl;
  }
  
}


void ModelBPR::updUIRatings(std::vector<std::tuple<int, int, int>>& bprTriplets, 
    const Data& data, Eigen::MatrixXf& T, int& subIter, int nTrainSamp, int start, int end) {

  double regMultNDiag =  (1.0 - 2.0*learnRate*l2Reg);
  double regMultDiag =  (1.0 - 2.0*learnRate*wl2Reg);
  int u, pI, nI;
  Eigen::VectorXf pdt(nFeatures);
  float r_ui, r_uj;
  
  for (int i = start; i < end; i++) {
    auto bprTriplet = bprTriplets[i];
    //get user, item and rating
    u    = std::get<0>(bprTriplet);
    pI   = std::get<1>(bprTriplet);
    nI   = std::get<2>(bprTriplet);
    
    r_ui = estPosRating(u, pI, data, pdt);
    r_uj = estNegRating(u, nI, data, pdt);
    double r_uij = r_ui - r_uj;
    double expCoeff = 1.0 /(1.0 + exp(r_uij));
    int locSubIter = subIter;
    //learnRate * expCoeff * f_u * f_i^T
    //lazySparseUpdMatWSpOuterPdt(W, T, data.uFAccumMat, u, data.itemFeatMat, pI, 
    //    learnRate*expCoeff, regMult, subIter, l1Reg);
    lazySparseUpdMatWSpOuterPdtD(W, T, data.uFAccumMat, u, data.itemFeatMat, pI, 
        learnRate*expCoeff, regMultDiag, regMultNDiag, locSubIter, wl1Reg, l1Reg);
    
    //- learnRate * expCoeff * f_u * f_j^T
    //lazySparseUpdMatWSpOuterPdt(W, T, data.uFAccumMat, u, data.itemFeatMat, nI, 
    //    -learnRate*expCoeff, regMult, subIter, l1Reg);
    lazySparseUpdMatWSpOuterPdtD(W, T, data.uFAccumMat, u, data.itemFeatMat, nI, 
        -learnRate*expCoeff, regMultDiag, regMultNDiag, locSubIter, wl1Reg, l1Reg);

    //-learnRate * expCoeff * f_i * f_i^T
    //lazySparseUpdMatWSpOuterPdt(W, T, data.itemFeatMat, pI, data.itemFeatMat, pI, 
    //    -learnRate*expCoeff, regMult, subIter, l1Reg);
    lazySparseUpdMatWSpOuterPdtD(W, T, data.itemFeatMat, pI, data.itemFeatMat, pI, 
        -learnRate*expCoeff, regMultDiag, regMultNDiag, locSubIter, wl1Reg, l1Reg);

    subIter++; 
    if (subIter >= nTrainSamp) {
      break;  
    }
  } 
}


void ModelBPR::parTrain(const Data &data, Model& bestModel) {

  std::cout << "\nModelBPR::parTrain" << std::endl;
  
  std::string prefix = "ModelBPR";

  load(prefix);

  int bestIter, u, pI, nI;
  Eigen::MatrixXf Wgrad(nFeatures, nFeatures);  
  Eigen::MatrixXf T(nFeatures, nFeatures);
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf jFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  float bestRecall, prevRecall;
 
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration; 
  
  start = std::chrono::system_clock::now();
  
  std::cout << "val recall: " <<  bestRecall << std::endl;
  end = std::chrono::system_clock::now();
  duration = end - start;
  std::cout << "\nValidation recall duration: " << duration.count() << std::endl;

  unsigned long const hwThreads = std::thread::hardware_concurrency();
  int nThreads = NTHREADS;
  if (hwThreads > 0  && hwThreads < NTHREADS) {
    nThreads = hwThreads;
  }

  std::cout << "nthreads: " << nThreads << std::endl;

  //random engine
  std::mt19937 mt(seed);
  
  auto uiRatings = getBPRUIRatings(data.trainMat);
  int nTrainSamp = uiRatings.size()*pcSamples;
  std::cout << "\nnBPR ratings: " << uiRatings.size() << " trainSamples: " << nTrainSamp << std::endl;

  std::vector<std::tuple<int, int, int>> bprTriplets;

  double regMultNDiag =  (1.0 - 2.0*learnRate*l2Reg);
  double regMultDiag =  (1.0 - 2.0*learnRate*wl2Reg);
  for (int iter = 0; iter < maxIter; iter++) {
    //shuffle the user item ratings
    std::shuffle(uiRatings.begin(), uiRatings.end(), mt);
    start = std::chrono::system_clock::now();
    T.fill(0);
    int subIter = 0;

    bprTriplets.clear();
    for (auto&& uiRating: uiRatings) {
      //get user, item and rating
      u    = std::get<0>(uiRating);
      pI   = std::get<1>(uiRating);
      //sample a negative item for user u
      nI = data.sampleNegItem(u);
      if (-1 == nI) {
        //failed to sample negativ item
        continue;
      }
      bprTriplets.push_back(std::make_tuple(u, pI, nI)); 
    }
    
    std::cout << "bprTriplets: " << bprTriplets.size() << std::endl;
    //call par methods on threads
    
    //allocate threads
    std::vector<std::thread> threads(nThreads-1);
    int nRatingsPerThread = bprTriplets.size()/nThreads;
    std::cout << "nRatingsPerThread: " << nRatingsPerThread << std::endl;
    for (int thInd = 0; thInd < nThreads-1; thInd++) {
      int startInd = thInd*nRatingsPerThread;
      int endInd = (thInd+1)*nRatingsPerThread;
      //std::cout << "thread: " << thInd << " " << startInd << " " << endInd << std::endl; 
      threads[thInd] = std::thread(&ModelBPR::updUIRatings, this, std::ref(bprTriplets),
          std::ref(data), std::ref(T), std::ref(subIter), nTrainSamp, 
          startInd, endInd);   
    
    } 

    //main thread
    int startInd = (nThreads-1)*nRatingsPerThread;
    int endInd = bprTriplets.size();
    //std::cout << "main thread: " << startInd << " " << endInd << std::endl;
    updUIRatings(bprTriplets, data, T, subIter, nTrainSamp, 
        startInd, endInd);

    //wait for threads to finish
    std::for_each(threads.begin(), threads.end(), 
        std::mem_fn(&std::thread::join));
     
    //perform reg updates on all the pairs
    for (int ind1 = 0; ind1 < nFeatures; ind1++) {
      for (int ind2 = 0; ind2 < nFeatures; ind2++) {
        if (ind1 == ind2) {
          //update with reg updates
          W(ind1, ind2) = W(ind1, ind2)*pow(regMultDiag, 
                                             subIter-T(ind1, ind2));
          //L1 or proximal update
          W(ind1, ind2) = proxL1(W(ind1, ind2), wl1Reg);
        } else {
          //update with reg updates
          W(ind1, ind2) = W(ind1, ind2)*pow(regMultNDiag, 
                                             subIter-T(ind1, ind2));
          //L1 or proximal update
          W(ind1, ind2) = proxL1(W(ind1, ind2), l1Reg);
        }
      }
    }


    end = std::chrono::system_clock::now();
    duration = end - start;
    auto subDuration = duration;

    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0 || iter == maxIter - 1) {
      start = std::chrono::system_clock::now();
      if(isTerminateModel(bestModel, data, iter, bestIter, bestRecall, 
          prevRecall)) {
        break;
      }
      end = std::chrono::system_clock::now();
      duration = end - start;
      std::cout << "\niter: " << iter << " val recall: " << prevRecall
        << " best recall: " << bestRecall << " subiter count: " << subIter 
        << " subiter duration: " << subDuration.count() 
        << " recall duration: " << duration.count() << std::endl;
      std::cout << "W norm: " << W.norm() << std::endl;

      bestModel.save(prefix);
    }
  
  }
  
}


void ModelBPR::train(const Data &data, Model& bestModel) {

  std::cout << "\nModelBPR::train" << std::endl;
  
  std::string prefix = "ModelBPR";

  load(prefix);

  int bestIter, u, pI, nI;
  Eigen::MatrixXf Wgrad(nFeatures, nFeatures);  
  Eigen::MatrixXf T(nFeatures, nFeatures);
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf jFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  Eigen::VectorXf pdt(nFeatures);
  float bestRecall, prevRecall, r_ui, r_uj;
  int trainNNZ = getNNZ(data.trainMat); 
 
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration; 
  
  start = std::chrono::system_clock::now();
  
  bestRecall = computeRecallParVec(data.valMat, data, 10, data.valItems);
  bestIter   = -1;
  bestModel  = *this;
  prevRecall = bestRecall;

  std::cout << "val recall: " <<  bestRecall << std::endl;
  end = std::chrono::system_clock::now();
  duration = end - start;
  std::cout << "\nValidation recall duration: " << duration.count() << std::endl;

  //random engine
  std::mt19937 mt(seed);
  
  auto uiRatings = getBPRUIRatings(data.trainMat);
  int nTrainSamp = uiRatings.size()*pcSamples;
  std::cout << "\nnBPR ratings: " << uiRatings.size() << " trainSamples: " << nTrainSamp << std::endl;

  double regMultNDiag =  (1.0 - 2.0*learnRate*l2Reg);
  double regMultDiag =  (1.0 - 2.0*learnRate*wl2Reg);
  for (int iter = 0; iter < maxIter; iter++) {
    //shuffle the user item ratings
    std::shuffle(uiRatings.begin(), uiRatings.end(), mt);
    start = std::chrono::system_clock::now();
    T.fill(0);
    int subIter = 0;
    for (auto&& uiRating: uiRatings) {
      //get user, item and rating
      u    = std::get<0>(uiRating);
      pI   = std::get<1>(uiRating);
      
      //sample a negative item for user u
      nI = data.sampleNegItem(u);
      
      if (-1 == nI) {
        //failed to sample negativ item
        continue;
      }

      r_ui = estPosRating(u, pI, data, pdt);
      r_uj = estNegRating(u, nI, data, pdt);
      double r_uij = r_ui - r_uj;
      double expCoeff = 1.0 /(1.0 + exp(r_uij));
 
      //learnRate * expCoeff * f_u * f_i^T
      //lazySparseUpdMatWSpOuterPdt(W, T, data.uFAccumMat, u, data.itemFeatMat, pI, 
      //    learnRate*expCoeff, regMult, subIter, l1Reg);
      lazySparseUpdMatWSpOuterPdtD(W, T, data.uFAccumMat, u, data.itemFeatMat, pI, 
          learnRate*expCoeff, regMultDiag, regMultNDiag, subIter, wl1Reg, l1Reg);
      
      //- learnRate * expCoeff * f_u * f_j^T
      //lazySparseUpdMatWSpOuterPdt(W, T, data.uFAccumMat, u, data.itemFeatMat, nI, 
      //    -learnRate*expCoeff, regMult, subIter, l1Reg);
      lazySparseUpdMatWSpOuterPdtD(W, T, data.uFAccumMat, u, data.itemFeatMat, nI, 
          -learnRate*expCoeff, regMultDiag, regMultNDiag, subIter, wl1Reg, l1Reg);

      //-learnRate * expCoeff * f_i * f_i^T
      //lazySparseUpdMatWSpOuterPdt(W, T, data.itemFeatMat, pI, data.itemFeatMat, pI, 
      //    -learnRate*expCoeff, regMult, subIter, l1Reg);
      lazySparseUpdMatWSpOuterPdtD(W, T, data.itemFeatMat, pI, data.itemFeatMat, pI, 
          -learnRate*expCoeff, regMultDiag, regMultNDiag, subIter, wl1Reg, l1Reg);

      subIter++; 
      if (subIter >= nTrainSamp) {
        break;  
      }
    } 
    
    //perform reg updates on all the pairs
    for (int ind1 = 0; ind1 < nFeatures; ind1++) {
      for (int ind2 = 0; ind2 < nFeatures; ind2++) {
        if (ind1 == ind2) {
          //update with reg updates
          W(ind1, ind2) = W(ind1, ind2)*pow(regMultDiag, 
                                             subIter-T(ind1, ind2));
          //L1 or proximal update
          W(ind1, ind2) = proxL1(W(ind1, ind2), wl1Reg);
        } else {
          //update with reg updates
          W(ind1, ind2) = W(ind1, ind2)*pow(regMultNDiag, 
                                             subIter-T(ind1, ind2));
          //L1 or proximal update
          W(ind1, ind2) = proxL1(W(ind1, ind2), l1Reg);
        }
      }
    }


    end = std::chrono::system_clock::now();
    duration = end - start;
    auto subDuration = duration;

    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0 || iter == maxIter - 1) {
      start = std::chrono::system_clock::now();
      if(isTerminateModelInit(bestModel, data, iter, bestIter, bestRecall, 
          prevRecall)) {
        break;
      }
      end = std::chrono::system_clock::now();
      duration = end - start;
      std::cout << "\niter: " << iter << " val recall: " << prevRecall
        << " best recall: " << bestRecall 
        << " subiter duration: " << subDuration.count() 
        << " recall duration: " << duration.count() << std::endl;
      std::cout << "W norm: " << W.norm() << std::endl;

      bestModel.save(prefix);
    }
  
  }
  
}


void ModelBPR::FTRLTrain(const Data &data, Model& bestModel) {

  std::cout << "\nModelBPR::FTRLTrain" << std::endl;
  
  std::string prefix = "ModelBPR";

  load(prefix);

  int bestIter, u, pI, nI;
  
  Eigen::MatrixXf Wgrad(nFeatures, nFeatures);  
  Eigen::MatrixXf z(nFeatures, nFeatures);
  Eigen::MatrixXf n(nFeatures, nFeatures);

  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf jFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  Eigen::VectorXf pdt(nFeatures);

  std::map<int, std::unordered_set<int>> coords;

  float bestRecall, prevRecall;
  int trainNNZ = getNNZ(data.trainMat); 
 
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> duration; 
  
  //random engine
  std::mt19937 mt(seed);
  
  auto uiRatings = getBPRUIRatings(data.trainMat);
  int nTrainSamp = uiRatings.size()*pcSamples;
  std::cout << "\nnBPR ratings: " << uiRatings.size() << " trainSamples: " << nTrainSamp << std::endl;

  z.fill(0);
  n.fill(0);
  float alpha = learnRate;
  float beta = learnRate;
  float lambda1, lambda2, coeff, signz;

  for (int iter = 0; iter < maxIter; iter++) {
    //shuffle the user item ratings
    std::shuffle(uiRatings.begin(), uiRatings.end(), mt);
    start = std::chrono::system_clock::now();
    int subIter = 0;
    for (auto&& uiRating: uiRatings) {
      //get user, item and rating
      u    = std::get<0>(uiRating);
      pI   = std::get<1>(uiRating);
      
      //sample a negative item for user u
      nI = data.sampleNegItem(u);
      
      if (-1 == nI) {
        //failed to sample negativ item
        continue;
      }


      computeBPRSparseGrad(u, pI, nI, Wgrad, pdt, data, coords);

      //go through the pairs
      for (auto&& kv: coords) {
        int ind1 = kv.first;
        for (auto&& ind2: kv.second) {
          if (ind1 == ind2) {
            lambda1 = wl1Reg;
            lambda2 = wl2Reg;
          } else {
            lambda1 = l1Reg;
            lambda2 = l2Reg;
          }

          if (z(ind1, ind2) >= -lambda1 && z(ind1, ind2) <= lambda1) {
            W(ind1, ind2) = 0;
          } else {
            coeff = -1.0/(((beta + std::sqrt(n(ind1, ind2)))/alpha) + lambda2);
            signz = -1;
            if (z(ind1, ind2) > 0) {
              signz = 1;
            }
            W(ind1, ind2) = coeff*(z(ind1, ind2) - signz*lambda1);
          }

        }
      }
      
      //go through the pairs
      for (auto&& kv: coords) {
        int ind1 = kv.first;
        for (auto&& ind2: kv.second) {
          //adagrad book keeping
          float sigma = (1.0/alpha)*(std::sqrt(n(ind1, ind2) + 
                Wgrad(ind1, ind2)*Wgrad(ind1, ind2)) - std::sqrt(n(ind1, ind2)));
          z(ind1, ind2) += Wgrad(ind1, ind2) - sigma*W(ind1, ind2);
          n(ind1, ind2) += Wgrad(ind1, ind2)*Wgrad(ind1, ind2);
        }
      }


      subIter++; 
      if (subIter >= nTrainSamp) {
        break;  
      }
    } 
    
    end = std::chrono::system_clock::now();
    duration = end - start;
    auto subDuration = duration;

    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0 || iter == maxIter - 1) {
      start = std::chrono::system_clock::now();
      if(isTerminateModel(bestModel, data, iter, bestIter, bestRecall, 
          prevRecall)) {
        break;
      }
      end = std::chrono::system_clock::now();
      duration = end - start;
      std::cout << "\niter: " << iter << " val recall: " << prevRecall
        << " best recall: " << bestRecall 
        << " subiter duration: " << subDuration.count() 
        << " recall duration: " << duration.count() << std::endl;
      std::cout << "W norm: " << W.norm() << std::endl;

      bestModel.save(prefix);
    }
  
  }
  
}
