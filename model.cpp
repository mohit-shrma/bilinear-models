#include "model.h"

Model::Model(const Params &params, int p_nFeatures) {
  nFeatures = p_nFeatures;
  l2Reg     = params.l2Reg;
  wReg      = params.wReg;
  nucReg    = params.nucReg;
  learnRate = params.learnRate;
  rank      = params.rank;
  maxIter   = params.maxIter;
  pcSamples = params.pcSamples; 
}


bool Model::isTerminateModel(Model& bestModel, const Data& data, int iter,
    int& bestIter, float& bestRecall, float& prevRecall) {
  
  bool ret = false;
  float currRecall = computeRecallPar(data.valMat, data, 10, data.valItems);
  
  if (iter > 0) {
    
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

  if (0 == iter) {
    bestModel = *this;
    bestRecall = currRecall;
    bestIter = iter;
  }

  prevRecall = currRecall;

  return ret;
}


bool Model::isTerminateModelObj(Model& bestModel, const Data& data, int iter,
    int& bestIter, float& bestObj, float& prevObj) {
  
  bool ret = false;
  float currObj = objective(data);
  
  if (iter > 0) {
    
    if (currObj < bestObj) {
      bestModel = *this;
      bestObj = currObj;
      bestIter = iter;
    }
    
    if (iter - bestIter >= CHANCE_ITER) {
      std::cout << "\nNOT CONVERGED: bestIter: " << bestIter << 
        " bestObj: " << bestObj << " currIter: " << iter << 
        " currObj: " << currObj << std::endl;
      ret = true;
    }

    if (fabs(prevObj - currObj) < EPS) {
      //convergence
      std::cout << "\nConverged in iteration: " << iter << " prevObj: "
        << prevObj << " currObj: " << currObj;
      ret = true;
    }

  }

  if (0 == iter) {
    bestModel = *this;
    bestObj = currObj;
    bestIter = iter;
  }

  prevObj = currObj;

  return ret;
}


float Model::computeRecall(gk_csr_t *mat, const Data &data, int N, 
    std::unordered_set<int> items) {

  int u, ii;
  int nRelevantUsers;
  float recall;

  //find whether users have items in test set
  std::vector<bool> isTestUser(mat->nrows, false);
  nRelevantUsers = 0;
  for (u = 0; u < mat->nrows; u++) {
    for (ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      if (mat->rowval[ii] > 0) {
        isTestUser[u] = true;
        nRelevantUsers++;
        break;
      }
    }
  } 
  
  //std::cout << "\nRelevant users with test items: " << nRelevantUsers;

  std::vector<float> uRecalls(mat->nrows);
  computeRecallUsers(mat, 0, mat->nrows, data, N, items, isTestUser, uRecalls);
  //compute ratings for each user on all test items and get top-N items
  
  recall = 0;
  for (u = 0; u < mat->nrows; u++) {
    if (uRecalls[u] >= 0) {
      recall += uRecalls[u];
    }
  }
  recall = recall/nRelevantUsers;
  
  return recall;
}


//compute recall from uStart(inclusive) to uEnd(exclusive), store results in
//uRecalls
void Model::computeRecallUsers(gk_csr_t *mat, int uStart, int uEnd, 
    const Data& data, int N, std::unordered_set<int>& items, 
    std::vector<bool>& isTestUser, std::vector<float>& uRecalls) {
  
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf pdt(nFeatures);
  auto comparePair = [](std::pair<int, float> a, std::pair<int, float> b) { 
    return a.second > b.second; 
  };
  float rating;
  std::unordered_set<int> topNitems;
  std::vector<std::pair<int, float>> itemRatings;
  itemRatings.reserve(items.size());
  int nItemsInTopN, nTestUserItems, testItem;
  float recall_u;

  int uCount = 0;
  
  //Eigen::MatrixXf uFeatWPdt = spMatMatPdt(data.itemFeatMat, W);

  for (int u = uStart; u < uEnd; u++) {
    
    if (!isTestUser[u]) {
      uRecalls[u] = -1;
      continue;
    }
    
    //compute ratings over all testItems
    itemRatings.clear();
    for (const int &item: items) {
      rating = estNegRating(u, item, data, pdt);
      itemRatings.push_back(std::make_pair(item, rating));
    }

    //put top-N item ratings pair in begining
    std::nth_element(itemRatings.begin(), itemRatings.begin()+N,
        itemRatings.end(), comparePair); 

    //get the set of top-N items for the user
    topNitems.clear();
    for (int i = 0; i < N; i++) {
      topNitems.insert(itemRatings[i].first);
    }

    nItemsInTopN = 0;
    nTestUserItems = 0;
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      testItem = mat->rowind[ii];
      nTestUserItems++;
      if (topNitems.find(testItem) != topNitems.end()) {
        //found test item
        nItemsInTopN++;
      }
    }

    if (nTestUserItems > N) {
      recall_u = (float)nItemsInTopN/(float)N;
    } else {
      recall_u = (float)nItemsInTopN/(float)nTestUserItems;
    }

    uRecalls[u] = recall_u;
    uCount++;
   
    /*
    if (uCount % 500 == 0) {
      std::cout << "\nustart: " << uStart << " uEnd: " << uEnd << " uCount: " 
        << uCount << std::endl;
    }
    */
    
  }
   
}


float Model::computeRecallPar(gk_csr_t *mat, const Data &data, int N, 
    std::unordered_set<int> items) {

  int u, i, ii;
  int nRelevantUsers;
  Eigen::VectorXf iFeat(nFeatures);

  //find whether users have items in test set
  std::vector<bool> isTestUser(mat->nrows, false);
  nRelevantUsers = 0;
  for (u = 0; u < mat->nrows; u++) {
    for (ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      if (mat->rowval[ii] > 0) {
        isTestUser[u] = true;
        nRelevantUsers++;
        break;
      }
    }
  } 
  
  unsigned long const hwThreads = std::thread::hardware_concurrency();
  //unsigned long const nThreads = std::min(hwThreads != 0? hwThreads:2, NTHREADS);
 
  //std::cout << "\nhwThreads: " << hwThreads;
  int nThreads = NTHREADS;
  if (hwThreads > 0  && hwThreads < NTHREADS) {
    nThreads = hwThreads;
  }
  //std::cout << "\nnthreads: " << nThreads;

  //allocate threads
  std::vector<std::thread> threads(nThreads-1);
  
  //storage for results from threads
  std::vector<float> uRecalls(mat->nrows);

  int nUsersPerThread = mat->nrows / nThreads;
  
  for (u = 0, i = 0; u < mat->nrows; u+=nUsersPerThread) {
    if (i < nThreads-1) {
      //start computation on thread
      threads[i++] = std::thread(&Model::computeRecallUsers, this, mat, u, 
          u+nUsersPerThread,
          std::ref(data), N, std::ref(items), std::ref(isTestUser), 
          std::ref(uRecalls));
    } else {
      //in main thread
      computeRecallUsers(mat, u, mat->nrows, data, N, items, isTestUser, 
          uRecalls);
      u = mat->nrows;
    }
  }
  
  //wait for threads to finish
  std::for_each(threads.begin(), threads.end(), 
      std::mem_fn(&std::thread::join));

  //std::cout << "\nRelevant users with test items: " << nRelevantUsers;

  //compute recall
  float recall = 0;
  for (u = 0; u < mat->nrows; u++) {
    if (uRecalls[u] >= 0) {
      recall += uRecalls[u];
    }
  }
  recall = recall/nRelevantUsers;
  
  return recall;
}


float Model::computeRMSE(gk_csr_t *mat, const Data& data) {
  
  int item;
  float r_ui, r_ui_est, rmse = 0;
  Eigen::VectorXf pdt(nFeatures);
  int nnz = 0;

  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      item = mat->rowind[ii];
      r_ui = mat->rowval[ii];
      r_ui_est = estPosRating(u, item, data, pdt);
      rmse += (r_ui - r_ui_est)*(r_ui - r_ui_est);
      nnz++;
    }
  }
  
  rmse = sqrt(rmse/nnz);
  return rmse;
}

int Model::invCount(std::vector<std::array<int,3>> sampTriplets, 
    const Data& data, Eigen::VectorXf& pdt) {
  
  int u, i, j;
  int invCount = 0;
  float r_ui, r_uj;

  for(auto&& triplet: sampTriplets) {
    u = triplet[0];
    i = triplet[1];
    j = triplet[2];
    r_ui = estPosRating(u, i, data, pdt);
    r_uj = estNegRating(u, j, data, pdt);
    if (r_ui < r_uj) {
      invCount++;
    }
  }
  return invCount;
}
