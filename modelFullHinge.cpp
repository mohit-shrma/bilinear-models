#include "modelFullHinge.h"


//return maximum negatively rated item
int ModelFullHinge::estRatingsforUser(int u, const Data& data, 
    std::map<int, float>& itemRatings) {
  int item, maxNegItem, maxNegRat;
  float r_ui, r_ui_est;
  gk_csr_t *trainMat = data.trainMat;
  Eigen::VectorXf pdt(nFeatures);
  
  maxNegItem = -1;
  maxNegRat = -1;
  itemRatings.clear();
  for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
    item = trainMat->rowind[ii];
    r_ui = trainMat->rowval[ii];
    if (r_ui < 1) {
      r_ui_est = estNegRating(u, item, data, pdt);
      itemRatings[item] = r_ui_est;
      if (-1 == maxNegItem) {
        maxNegItem = item;
        maxNegRat = r_ui_est;
      } else if (maxNegRat < r_ui_est) {
        maxNegItem = item;
        maxNegRat = r_ui_est;
      }
    } else {
      r_ui_est = estPosRating(u, item, data, pdt); 
      itemRatings[item] = r_ui_est;
    }
  }

  return maxNegItem;
}


int ModelFullHinge::computeGrad(int u, Eigen::MatrixXf& Wgrad, 
    Eigen::MatrixXf& gradNegHull, const Data& data, int maxNegItem, 
    std::map<int, float>& itemRatings) {
  int ii, item, maxNegItemCount, posItemCount, updItemCount;
  gk_csr_t* mat = data.trainMat;
  float maxNegRat, r_ui, r_ui_est;
  
  //get max rating item
  maxNegRat = itemRatings[maxNegItem];

  //compute convex hull of max neg rated items
  gradNegHull.fill(0);
  maxNegItemCount = 0;
  Wgrad.fill(0);
  posItemCount = 0;
  updItemCount = 0;
  for (ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
    item = mat->rowind[ii];
    r_ui = mat->rowval[ii];
    r_ui_est = itemRatings[item];
    if (r_ui < 1) {
      //-ve item found
      if (r_ui_est == maxNegRat) {
        //f_u*f_j^T
        updateMatWSpOuterPdt(gradNegHull, data.uFAccumMat, u, data.itemFeatMat,
            item, 1); 
        maxNegItemCount++;
      }
    }  else {
      //positive rated item
      //grad if r_ui_est < r_ujmax + 1
      if (r_ui_est < maxNegRat + 1) {
        //-f_u*f_i^T
        updateMatWSpOuterPdt(Wgrad, data.uFAccumMat, u, data.itemFeatMat, item, -1);
        //f_i*f_i^T
        updateMatWSpOuterPdt(Wgrad, data.itemFeatMat, item, data.itemFeatMat, item, 1);
        //add convex hull of neg items to gradient
        //Wgrad += gradNegHull;
        updItemCount++;
      }
      posItemCount++;
    }
  }
  
  gradNegHull = gradNegHull/maxNegItemCount;

  Wgrad += updItemCount*gradNegHull;
  Wgrad = Wgrad/posItemCount;
  Wgrad += l2Reg*2*W;

  return updItemCount;
}


void ModelFullHinge::train(const Data& data, Model& bestModel) {

  std::cout << "\nModelFullHinge::train" << std::endl;

  int bestIter, u;
  Eigen::MatrixXf Wgrad(nFeatures, nFeatures);  
  Eigen::MatrixXf gradNegHull(nFeatures, nFeatures);  
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf jFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  Eigen::VectorXf pdt(nFeatures);
  float bestRecall, prevRecall;
  int trainNNZ = getNNZ(data.trainMat); 
 
  std::cout << "\ntrain nnz: " << trainNNZ << " trainSamples: " << trainNNZ*pcSamples << std::endl;
  //std::cout << "val recall: " << computeRecallPar(data.valMat, data, 10, data.valItems) << std::endl;

  std::map<int, float> itemRatings;
  std::map<int, int> uUpdCountMap;
  int maxNegItem, uUpdCount;
  std::set<int> invalidUsers;

  std::chrono::time_point<std::chrono::system_clock> start, end;
  for (int iter = 0; iter < maxIter; iter++) {
    start = std::chrono::system_clock::now();
    //for (int subIter = 0; subIter < trainNNZ*pcSamples; subIter++) {
    int uCount = 0;
    for (int subIter = 0; subIter < 2*data.nUsers; subIter++) {
        
      //sample user
      while (1) {
        u = std::rand() % data.nUsers;
        auto search = invalidUsers.find(u);
        if (search != invalidUsers.end()) {
          //found u in invalidUsers
          continue;
        }
        if (data.posTrainUsers.find(u) != data.posTrainUsers.end()) {
          //found u
          break;
        }
      }

      //estimate ratings for the item rated by user and get the maxNegItem
      maxNegItem = estRatingsforUser(u, data, itemRatings);
      if (-1 == maxNegItem) {
        uCount++;
        invalidUsers.insert(u);
        continue;
      }


      //compute gradient
      //gradCheck(uFeat, iFeat, jFeat, Wgrad); 
      //computeBPRGrad(uFeat, iFeat, jFeat, Wgrad);
      uUpdCount = computeGrad(u, Wgrad, gradNegHull, data, maxNegItem, itemRatings);
      
      uUpdCountMap[u] = uUpdCount;

      //update W
      W -= learnRate*Wgrad;
    } 
  
    int invCount = 0;
    for (auto it = uUpdCountMap.begin(); it != uUpdCountMap.end(); ++it) {
      invCount += it->second;
    }

    std::cerr << "\nNo max -ve item for users: " << invalidUsers.size()
       << " uCount: " << uCount << " invCount: "<< invCount 
       << " W norm: " << W.norm() << std::endl;
    
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = end - start;
    
    //nuclear norm projection after all sub-iters
    //performNucNormProjSVDLib(W, rank);
    
    //perform model evaluation on validation set
    if (iter %OBJ_ITER == 0) {
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




