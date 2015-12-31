#include "util.h"


std::unordered_set<int> getPosUsers(gk_csr_t *mat) {
  std::unordered_set<int> userSet;
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      if (mat->rowval[ii] > 0) {
        userSet.insert(u);
        break;
      }
    }
  }

  return userSet;
}


std::unordered_set<int> getItemSet(gk_csr_t *mat) {
  
  std::unordered_set<int> itemSet;
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      itemSet.insert(mat->rowind[ii]);
    }
  }
  
  return itemSet;
}


int getNNZ(gk_csr_t *mat) {
  int nnz = 0;
  for (int u = 0; u < mat->nrows; u++) {
    nnz += mat->rowptr[u+1] - mat->rowptr[u];
  }
  return nnz;
}


void extractFeat(gk_csr_t *itemFeatMat, int item, Eigen::VectorXf& fVec) {
  
  //init fVec to 0
  fVec.fill(0);
  for (int ii = itemFeatMat->rowptr[item]; ii < itemFeatMat->rowptr[item+1];
        ii++) {
    fVec[itemFeatMat->rowind[ii]] = itemFeatMat->rowval[ii];
  }

}

