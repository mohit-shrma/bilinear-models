#include "model.h"

//estimate rating on a positively rated item
float Model::estPosRating(int u, int item, const Data& data,
    Eigen::VectorXf& pdt) {
  float r_ui = 0;
  pdt.fill(0);

  //compute dot product of mat and sparse vector 
  matSpVecPdt(W, data.itemFeatMat, item, pdt);
  //f_u^TWf_i 
  r_ui = vecSpVecDot(pdt, data.uFAccumMat, u);
  //-f_i^TWf_i
  r_ui -= vecSpVecDot(pdt, data.itemFeatMat, item);

  return r_ui; 
} 


//estimate rating on a item not rated before
float Model::estNegRating(int u, int item, const Data& data, 
    Eigen::VectorXf& pdt) {
  float r_ui = 0;
  pdt.fill(0);

  //compute dot product of mat and sparse vector 
  matSpVecPdt(W, data.itemFeatMat, item, pdt);
  r_ui = vecSpVecDot(pdt, data.uFAccumMat, u);
 
  return r_ui;
}


Model::Model(const Params &params, int p_nFeatures) {
  nFeatures = p_nFeatures;
  l2Reg     = params.l2Reg;
  nucReg    = params.nucReg;
  learnRate = params.learnRate;
  rank      = params.rank;
  maxIter   = params.maxIter;
  pcSamples = params.pcSamples; 
  //initialize model matrix
  W = Eigen::MatrixXf::Zero(nFeatures, nFeatures);
  for (int i = 0; i < nFeatures; i++) {
    for (int j = 0; j < nFeatures; j++) {
      W(i,j) = (float)std::rand()/ (float)(1.0 + RAND_MAX);
    }
  }
  
  std::cout << "\nW norm: " << W.norm() << std::endl;
}


bool Model::isTerminateModel(Model& bestModel, const Data& data, int iter,
    int& bestIter, float& bestRecall, float& prevRecall) {
  
  bool ret = false;
  float currRecall = computeRecall(data.valMat, data, 10, data.valItems);
  
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
        << prevRecall << " currRecall: " << currRecall << " W norm: "
        << W.norm();
      ret = true;
    }

  }

  if (0 == iter) {
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
    bestObj = currObj;
    bestIter = iter;
  }

  prevObj = currObj;

  return ret;
}


float Model::computeRecall(gk_csr_t *mat, const Data &data, int N, 
    std::unordered_set<int> items) {

  int u, i, ii;
  int nRelevantUsers;
  float rating;
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf pdt(nFeatures);

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

  auto comparePair = [](std::pair<int, float> a, std::pair<int, float> b) { 
    return a.second > b.second; 
  };
  std::unordered_set<int> topNitems;
  std::vector<std::pair<int, float>> itemRatings;
  std::vector<float> uRecalls(mat->nrows);
  itemRatings.reserve(items.size());
  int nItemsInTopN, nTestUserItems, testItem;
  float recall, recall_u;
  recall = 0;
  //compute ratings for each user on all test items and get top-N items
  for (u = 0; u < mat->nrows; u++) {
    if (!isTestUser[u]) {
      uRecalls[u] = -1;
      continue;
    }
    //compute ratings over all testItems
    for (const int &item: items) {
      //extractFeat(data.itemFeatMat, item, iFeat);
      //rating = data.uFeatAcuum.row(u)*W*iFeat;
      matSpVecPdt(W, data.itemFeatMat, item, pdt);
      //rating = pdt.dot(data.uFeatAcuum.row(u));
      rating = vecSpVecDot(pdt, data.uFAccumMat, u);
      itemRatings.push_back(std::make_pair(item, rating));
    }
    
    //put top-N item ratings pair in begining
    std::nth_element(itemRatings.begin(), itemRatings.begin()+N,
        itemRatings.end(), comparePair); 
    
    //get the set of top-N items for the user
    topNitems.clear();
    for (i = 0; i < N; i++) {
      topNitems.insert(itemRatings[i].first);
    }
    
    nItemsInTopN = 0;
    nTestUserItems = 0;
    for (ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
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
      
    if (0 == u%500) {
      std::cout << "\nu = " << u  << " " << recall_u << std::endl;
    }

    uRecalls[u] = recall_u;

    recall += recall_u;
  }
  writeFVec(uRecalls, "ser_recall.txt"); 
  recall = recall/nRelevantUsers;
  
  return recall;
}


//compute recall from uStart(inclusive) to uEnd(exclusive), store results in
//uRecalls
void Model::computeRecallUsers(gk_csr_t *mat, int uStart, int uEnd, 
    const Data& data, int N, std::unordered_set<int>& items, 
    std::vector<bool>& isTestUser, std::vector<float>& uRecalls) {
  
  Eigen::VectorXf iFeat(nFeatures);
  auto comparePair = [](std::pair<int, float> a, std::pair<int, float> b) { 
    return a.second > b.second; 
  };
  float rating;
  std::unordered_set<int> topNitems;
  std::vector<std::pair<int, float>> itemRatings;
  itemRatings.reserve(items.size());
  int nItemsInTopN, nTestUserItems, testItem;
  float recall_u;

  for (int u = uStart; u < uEnd; u++) {
    
    if (!isTestUser[u]) {
      uRecalls[u] = -1;
      continue;
    }
    
    //compute ratings over all testItems
    for (const int &item: items) {
      extractFeat(data.itemFeatMat, item, iFeat);
      rating = data.uFeatAcuum.row(u)*W*iFeat;
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
 
  std::cout << "\nhwThreads: " << hwThreads;
  int nThreads = NTHREADS;
  if (hwThreads > 0  && hwThreads < NTHREADS) {
    nThreads = hwThreads;
  }
  std::cout << "\nnthreads: " << nThreads;

  //allocate threads
  std::vector<std::thread> threads(nThreads-1);
  
  //storage for results from threads
  std::vector<float> uRecalls(mat->nrows);

  int nUsersPerThread = mat->nrows / nThreads;
  
  for (u = 0, i = 0; u < mat->nrows; u+=nUsersPerThread) {
    if (u < mat->nrows-nUsersPerThread) {
      threads[i++] = std::thread(&Model::computeRecallUsers, this, mat, u, 
          u+nUsersPerThread,
          std::ref(data), N, std::ref(items), std::ref(isTestUser), 
          std::ref(uRecalls));
    } else {
      //in main thread
      computeRecallUsers(mat, u, mat->nrows, data, N, items, isTestUser, 
          uRecalls);
    }
  }
  //wait for threads to finish
  std::for_each(threads.begin(), threads.end(), 
      std::mem_fn(&std::thread::join));

  std::cout << "\nRelevant users with test items: " << nRelevantUsers;

  writeFVec(uRecalls, "par_recall.txt"); 
  //compute recall
  float recall = 0;
  for (u = 0; u < mat->nrows; u++) {
    if (uRecalls[u] >= 0) {
      recall += uRecalls[u];
    }
  }
  recall = recall/mat->nrows;
  
  return recall;
}


float Model::computeRMSE(gk_csr_t *mat, const Data& data) {
  
  int item;
  float r_ui, r_ui_est, rmse = 0;
  Eigen::VectorXf iFeat(nFeatures);
  Eigen::VectorXf uFeat(nFeatures);
  int nnz = 0;

  for (int u = 0; u < mat->nrows; u++) {
    uFeat = data.uFeatAcuum.row(u);
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      item = mat->rowind[ii];
      extractFeat(data.itemFeatMat, item, iFeat);
      r_ui = mat->rowval[ii];
      r_ui_est = (uFeat - iFeat).transpose()*W*iFeat;
      rmse += (r_ui - r_ui_est)*(r_ui - r_ui_est);
      nnz++;
    }
  }
  
  rmse = sqrt(rmse/nnz);
  return rmse;
}

