#ifndef _DATA_STRUCT_H_
#define _DATA_STRUCT_H_

#include <iostream>
#include <unordered_set>
#include "util.h"
#include "GKlib.h"

class Params {

  public:
    char *trainMatFile;
    char *testMatFile;
    char *valMatFile;
    char *itemFeatureFile;

    float l2Reg;
    float nucReg;
    float learnRate;
    int rank;

    int seed; 
    
    float pcSamples; 
    int maxIter; 
    bool isFeatNorm;

    Params(char *p_trainMatFile, char *p_testMatFile, char *p_valMatFile,
            char *p_itemFeatureFile, 
            float p_l2Reg, float p_nucReg, float p_learnRate, int p_rank,
            int p_seed, float p_pcSamples, int p_maxIter, bool p_isFeatNorm)
            : trainMatFile(p_trainMatFile), testMatFile(p_testMatFile),
            valMatFile(p_valMatFile), itemFeatureFile(p_itemFeatureFile),
            l2Reg(p_l2Reg), nucReg(p_nucReg), learnRate(p_learnRate),
            rank(p_rank), seed(p_seed), pcSamples(p_pcSamples),
            maxIter(p_maxIter), isFeatNorm(p_isFeatNorm) {}
    
    void display() {
      std::cout << "\ntrainMatFile: " << trainMatFile;
      std::cout << "\ntestMatFile: " << testMatFile;
      std::cout << "\nvalMatFile: " << valMatFile;
      std::cout << "\nitemFeatureFile: " << itemFeatureFile;
      std::cout << "\nl2Reg: " << l2Reg;
      std::cout << "\nnucReg: " << nucReg;
      std::cout << "\nlearnRate: " << learnRate;
      std::cout << "\nrank: " << rank;
      std::cout << "\nseed: " << seed;
      std::cout << "\npcSamples: " << pcSamples;
      std::cout << "\nmaxIter: " << maxIter;
      std::cout << "\nisFeatNorm: " << isFeatNorm;
    }

};


class Data {
  
  public:
    int nFeatures;
    int nUsers; 
    gk_csr_t *trainMat;
    gk_csr_t *testMat;
    gk_csr_t *valMat;
    gk_csr_t *itemFeatMat;
     
    Eigen::MatrixXf uFeatAcuum; 
    std::unordered_set<int> testItems;
    std::unordered_set<int> valItems;
    std::unordered_set<int> trainItems;
    std::unordered_set<int> posTrainUsers;

    Data(const Params& params) {
      trainMat = NULL;
      testMat = NULL;
      valMat = NULL;
      itemFeatMat = NULL;

      if (NULL != params.trainMatFile) {
        std::cout << "\nReading partial train matrix 0-indexed... ";
        trainMat = gk_csr_Read(params.trainMatFile, GK_CSR_FMT_CSR, 1, 0);
        gk_csr_CreateIndex(trainMat, GK_CSR_COL);
        nUsers = trainMat->nrows;
      }
      
      if (NULL != params.testMatFile) {
        std::cout << "\nReading partial test matrix 0-indexed... ";
        testMat = gk_csr_Read(params.testMatFile, GK_CSR_FMT_CSR, 1, 0);
        gk_csr_CreateIndex(testMat, GK_CSR_COL);
      }
      
      if (NULL != params.valMatFile) {
        std::cout << "\nReading partial val matrix 0-indexed... ";
        valMat = gk_csr_Read(params.valMatFile, GK_CSR_FMT_CSR, 1, 0);
        gk_csr_CreateIndex(valMat, GK_CSR_COL);
      }
      
      if (NULL != params.itemFeatureFile) {
        std::cout << "\nRading item-features matrix 0-indexed... ";
        itemFeatMat = gk_csr_Read(params.itemFeatureFile, GK_CSR_FMT_CSR, 1, 0);
        if (params.isFeatNorm) {
          gk_csr_Normalize(itemFeatMat, GK_CSR_ROW, 2);
        }
        gk_csr_CreateIndex(itemFeatMat, GK_CSR_COL);
        nFeatures = itemFeatMat->ncols;
      }
      
      uFeatAcuum = Eigen::MatrixXf::Zero(nUsers, nFeatures);
      //accumulate all user features
      for (int u = 0; u < nUsers; u++) {
        for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1];
              ii++) {
          int itemInd = trainMat->rowind[ii];
          for (int k = itemFeatMat->rowptr[itemInd];
               k < itemFeatMat->rowptr[itemInd+1]; k++) {
            //NOTE: multiplying by rating to make sure explicit '0' excluded
            uFeatAcuum(u, itemFeatMat->rowind[k]) += trainMat->rowval[u]*itemFeatMat->rowval[k];
          }
        }
      }
    
      testItems     = getItemSet(testMat);
      trainItems    = getItemSet(trainMat);
      valItems      = getItemSet(valMat);
      posTrainUsers = getPosUsers(trainMat);
    }
    

    ~Data() {
      if (trainMat) {
        gk_csr_Free(&trainMat);
      }
      if (testMat) {
        gk_csr_Free(&testMat);
      }
      if (valMat) {
        gk_csr_Free(&valMat);
      }
      if (itemFeatMat) {
        gk_csr_Free(&itemFeatMat);
      }
      //TODO: make sure eigen matrix freed
    }

};


#endif


