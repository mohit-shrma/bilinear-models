#ifndef _DATA_STRUCT_H_
#define _DATA_STRUCT_H_

#include <iostream>
#include <fstream>
#include <unordered_set>
#include <array>
#include "io.h"
#include "util.h"
#include "GKlib.h"
#include "const.h"

class Params {

  public:
    char *trainMatFile;
    char *testMatFile;
    char *valMatFile;
    char *itemFeatureFile;
    char *featAccumFile;
    
    float l1Reg; //regularization for non-diag component
    float l2Reg; //regularization for non-diag component 
    float wl1Reg; //regularization for linear/diag component
    float wl2Reg; //regularization for linear/diag component
    float nucReg;
    float learnRate;
    int rank;

    int seed; 
    
    float pcSamples; 
    int maxIter; 
    bool isFeatNorm;

    Params(char *p_trainMatFile, char *p_testMatFile, char *p_valMatFile,
            char *p_itemFeatureFile, char *p_featAccumFile, float p_l1Reg, 
            float p_l2Reg, float p_wl1Reg, float p_wl2Reg, float p_nucReg, 
            float p_learnRate, int p_rank, int p_seed, float p_pcSamples, 
            int p_maxIter, bool p_isFeatNorm)
            : trainMatFile(p_trainMatFile), testMatFile(p_testMatFile), 
            valMatFile(p_valMatFile), itemFeatureFile(p_itemFeatureFile),
            featAccumFile(p_featAccumFile), l1Reg(p_l1Reg), 
            l2Reg(p_l2Reg), wl1Reg(p_wl1Reg), wl2Reg(p_wl2Reg), nucReg(p_nucReg), 
            learnRate(p_learnRate),rank(p_rank), seed(p_seed), 
            pcSamples(p_pcSamples), maxIter(p_maxIter), isFeatNorm(p_isFeatNorm) {}
    
    void display() {
      std::cout << "\n****************** INPUT **************";
      std::cout << "\ntrainMatFile: " << trainMatFile;
      std::cout << "\ntestMatFile: " << testMatFile;
      std::cout << "\nvalMatFile: " << valMatFile;
      std::cout << "\nitemFeatureFile: " << itemFeatureFile;
      std::cout << "\nfeatAccumFile: " << featAccumFile;
      
      std::cout << "\n****************** PARAMETERS **************";
      std::cout << "\nl1Reg: " << l1Reg;
      std::cout << "\nl2Reg: " << l2Reg;
      std::cout << "\nwl1Reg: " << wl1Reg;
      std::cout << "\nwl2Reg: " << wl2Reg;
      std::cout << "\nnucReg: " << nucReg;
      std::cout << "\nlearnRate: " << learnRate;
      std::cout << "\nrank: " << rank;

      std::cout << "\nseed: " << seed;
      
      std::cout << "\npcSamples: " << pcSamples;
      std::cout << "\nmaxIter: " << maxIter;
      std::cout << "\nisFeatNorm: " << isFeatNorm;
      std::cout << "\n********************************************" << std::endl;
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
    gk_csr_t *uFAccumMat;

    std::unordered_set<int> testItems;
    std::unordered_set<int> valItems;
    std::unordered_set<int> trainItems;
    std::unordered_set<int> posTrainUsers;

    void printDetails() {
      std::cout << "\nnUsers: " << nUsers << " nFeatures: " << nFeatures;
      std::cout << "\ntrain nrows: " << trainMat->nrows << " ncols: "  
        << trainMat->ncols;
      std::cout << "\ntest nrows: " << testMat->nrows << " ncols: "  
        << testMat->ncols;
      std::cout << "\nval nrows: " << valMat->nrows << " ncols: "  
        << valMat->ncols;
      std::cout << "\nitems: " << itemFeatMat->nrows << " nFeat: " << itemFeatMat->ncols;
      std::cout << "\nusers: " << uFAccumMat->nrows << " nFeat: " << uFAccumMat->ncols;
      std::cout << "\nnTestItems: " << testItems.size();
      std::cout << "\nnValItems: " << valItems.size();
      std::cout << "\ntrainItems: " << trainItems.size() << std::endl;
      int itemFeatNNz = getNNZ(itemFeatMat);
      std::cout << "\navg item feat/row: " << (double)itemFeatNNz / (double) itemFeatMat->nrows;
      std::cout << "\navg item feat/col: " << (double)itemFeatNNz / (double) itemFeatMat->ncols;
      int uFeatNNZ = getNNZ(uFAccumMat);
      std::cout << "\navg u feat/row: " << (double)uFeatNNZ / (double) uFAccumMat->nrows;
      std::cout << "\navg u feat/col: " << (double)uFeatNNZ / (double) uFAccumMat->ncols;
      std::cout << "\nnposTrainUsers: " << posTrainUsers.size() << std::endl;
    }


    Data(const Params& params) {
      trainMat = NULL;
      testMat = NULL;
      valMat = NULL;
      itemFeatMat = NULL;

      if (NULL != params.trainMatFile) {
        if (!isFileExist(params.trainMatFile)) {
          std::cout << "\n!!!Train file NOT FOUND!!!" << std::endl;
          exit(1);
        }
        std::cout << "\nReading partial train matrix 1-indexed... " << std::endl;
        trainMat = gk_csr_Read(params.trainMatFile, GK_CSR_FMT_CSR, 1, CSR1INDEXED);
        gk_csr_CreateIndex(trainMat, GK_CSR_COL);
        nUsers = trainMat->nrows;
      }
      
      if (NULL != params.testMatFile) {
        if (!isFileExist(params.testMatFile)) {
          std::cout << "\n!!!Test file NOT FOUND!!!" << std::endl;
          exit(1);
        }
        std::cout << "\nReading partial test matrix 1-indexed... " << std::endl;
        testMat = gk_csr_Read(params.testMatFile, GK_CSR_FMT_CSR, 1, CSR1INDEXED);
        gk_csr_CreateIndex(testMat, GK_CSR_COL);
      }
      
      if (NULL != params.valMatFile) {
        if (!isFileExist(params.valMatFile)) {
          std::cout << "\n!!!Val file NOT FOUND!!!" << std::endl;
          exit(1);
        }
        std::cout << "\nReading partial val matrix 1-indexed... " << std::endl;
        valMat = gk_csr_Read(params.valMatFile, GK_CSR_FMT_CSR, 1, CSR1INDEXED);
        gk_csr_CreateIndex(valMat, GK_CSR_COL);
      }
      
      if (NULL != params.itemFeatureFile) {
        if (!isFileExist(params.itemFeatureFile)) {
          std::cout << "\n!!!Item feature file NOT FOUND!!!" << std::endl;
          exit(1);
        }
        std::cout << "\nReading item-features matrix 1-indexed... " << std::endl;
        itemFeatMat = gk_csr_Read(params.itemFeatureFile, GK_CSR_FMT_CSR, 1, CSR1INDEXED);
        if (params.isFeatNorm) {
          std::cout << "\nNormalizing item features...";
          gk_csr_Normalize(itemFeatMat, GK_CSR_ROW, 2);
        }
        gk_csr_CreateIndex(itemFeatMat, GK_CSR_COL);
        nFeatures = itemFeatMat->ncols;
      }
     
      if (NULL != params.featAccumFile) {
        if (!isFileExist(params.featAccumFile)) {
          std::cout << "\n!!!User accumulation file NOT FOUND!!!" << std::endl;
          exit(1);
          /* 
          std::ofstream opMat(params.featAccumFile);
          if (opMat.is_open()) {
            Eigen::MatrixXf uFeatAcuum = Eigen::MatrixXf::Zero(nUsers, nFeatures);
            int tempNNZ = 0;
            std::cout << "\nWriting file containing user accumulated featues..."
              << std::endl;
            //accumulate all user features
            for (int u = 0; u < nUsers; u++) {
              for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1];
                    ii++) {
                int item = trainMat->rowind[ii];
                float itemRating = trainMat->rowval[ii];
                for (int k = itemFeatMat->rowptr[item];
                     k < itemFeatMat->rowptr[item+1]; k++) {
                  //NOTE: multiplying by rating to make sure explicit '0' excluded
                  uFeatAcuum(u, itemFeatMat->rowind[k]) += itemRating*itemFeatMat->rowval[k];
                }
              }
              
              for (int k = 0; k < nFeatures; k++) {
                if (uFeatAcuum(u,k) > 0) {
                  opMat << k+1 << " " << uFeatAcuum(u, k) << " ";
                  tempNNZ++;
                }
              }

              opMat << std::endl;
            }

            //close op file
            opMat.close();
            
            std::cout << "\nuFeatAccum NNZ: " << tempNNZ <<  " density: "
              << (float)tempNNZ/float(nUsers*nFeatures)  << std::endl;
          }
          */
           
        }         

        std::cout << "\nReading accumulated features matrix 1-indexed... " << std::endl;
        uFAccumMat = gk_csr_Read(params.featAccumFile, GK_CSR_FMT_CSR, 1, 1);
        gk_csr_CreateIndex(uFAccumMat, GK_CSR_COL);
        std::cout << "\nnnz ufaccum: " << getNNZ(uFAccumMat);
      }

      testItems     = getItemSet(testMat);
      trainItems    = getItemSet(trainMat);
      valItems      = getItemSet(valMat);
      posTrainUsers = getPosUsers(trainMat);
      printDetails();
    }
    

  std::array<int, 3> sampleTriplet() const {
    
    std::array<int, 3> triplet{{-1, -1, -1}};
    int u = -1, i = -1, j = -1, ii, jj;
    int nUserItems, start, end;
    int nTrainItems = trainItems.size();
    int32_t *ui_rowind = trainMat->rowind;
    ssize_t *ui_rowptr = trainMat->rowptr;
    float   *ui_rowval = trainMat->rowval;
    
    //sample user
    while (1) {
      u = std::rand() % nUsers;
      if (posTrainUsers.find(u) != posTrainUsers.end()) {
        //found u
        break;
      }
    }
    triplet[0] = u;

    //sample pos item
    nUserItems = ui_rowptr[u+1] - ui_rowptr[u];
    while (1) {
      ii = std::rand()%nUserItems + ui_rowptr[u];
      i = ui_rowind[ii];
      if (ui_rowval[ii] > 0) {
        break;
      }
    }
    triplet[1] = i;

    //sample neg item
    while(1) {
      jj = std::rand()%nUserItems;
      if (ui_rowval[jj + ui_rowptr[u]] == 0.0) {
        //explicit 0
        j = ui_rowind[jj + ui_rowptr[u]];
        break;
      } else {
        //search for implicit 0
        
        if (0 == jj) {
          start = 0;
          end = ui_rowind[ui_rowptr[u]]; //first rated item by u
        } else if (nUserItems-1 == jj) {
          start = ui_rowind[ui_rowptr[u] + jj] + 1; //item next to last rated item
          end = nTrainItems;
        } else {
          start = ui_rowind[ui_rowptr[u] + jj] + 1; //item next to jjth item
          end = ui_rowind[ui_rowptr[u] + jj + 1]; //item rated after jjth item
        }

        //check for empty interval
        if (end - start > 0) {
          j = std::rand()%(end-start) + start;
        } else {
          continue;
        }

        //make sure sampled -ve item not present in testSet and valSet
        if (testItems.find(j) != testItems.end() ||
            valItems.find(j) != valItems.end()) {
          //found in either set
          continue;
        }

        if (trainItems.find(j) != trainItems.end()) {
          break;
        }
      }
    } //end while

    triplet[2] = j;

    return triplet;
  }
  

  int sampleNegItem(int u) const {
    
    int j = -1, jj;
    int nUserItems, start, end;
    int nTrainItems = trainItems.size();
    int32_t *ui_rowind = trainMat->rowind;
    ssize_t *ui_rowptr = trainMat->rowptr;
    float   *ui_rowval = trainMat->rowval;
    int nTry = 0; 
    nUserItems = ui_rowptr[u+1] - ui_rowptr[u];
    //sample neg item
    while(nTry < 100) {
      jj = std::rand()%nUserItems;
      if (ui_rowval[jj + ui_rowptr[u]] == 0.0) {
        //explicit 0
        j = ui_rowind[jj + ui_rowptr[u]];
        break;
      } else {
        //search for implicit 0
        
        if (0 == jj) {
          start = 0;
          end = ui_rowind[ui_rowptr[u]]; //first rated item by u
        } else if (nUserItems-1 == jj) {
          start = ui_rowind[ui_rowptr[u] + jj] + 1; //item next to last rated item
          end = nTrainItems;
        } else {
          start = ui_rowind[ui_rowptr[u] + jj] + 1; //item next to jjth item
          end = ui_rowind[ui_rowptr[u] + jj + 1]; //item rated after jjth item
        }

        //check for empty interval
        if (end - start > 0) {
          j = std::rand()%(end-start) + start;
        } else {
          continue;
        }

        //make sure sampled -ve item not present in testSet and valSet
        if (testItems.find(j) != testItems.end() ||
            valItems.find(j) != valItems.end()) {
          //found in either set
          continue;
        }

        if (trainItems.find(j) != trainItems.end()) {
          break;
        }
      }
      nTry++;
    } //end while
    
    if (100 == nTry) {
      j = -1;
    }

    return j;
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
      if (uFAccumMat) {
        gk_csr_Free(&uFAccumMat);
      }
      //TODO: make sure eigen matrix freed
    }

};


#endif


