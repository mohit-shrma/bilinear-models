#include "util.h"

//get users who have atleast one greater than zero rating
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


std::vector<std::tuple<int, int, float>> getUIRatings(gk_csr_t* mat) {
  std::vector<std::tuple<int, int, float>> uiRatings;
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      float rating = mat->rowval[ii];
      uiRatings.push_back(std::make_tuple(u, item, rating));
    }
  }
  return uiRatings;
}


std::vector<std::tuple<int, int, float>> getBPRUIRatings(gk_csr_t* mat) {
  std::vector<std::tuple<int, int, float>> uiRatings;
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      float rating = mat->rowval[ii];
      if (rating > 0) {
        uiRatings.push_back(std::make_tuple(u, item, rating));
      }
    }
  }
  return uiRatings;
}


void matStats(Eigen::MatrixXf& mat) {
  int nrows = mat.rows();
  int ncols = mat.cols();
  std::cout << "nrows: " << nrows << " ncols: " << ncols << std::endl;
  int nnz = 0;
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      if (fabs(mat(i,j)) >= EPS) {
        nnz++;
      }
    }
  }  
  std::cout << "nnz: " << nnz << " density: " << (float)nnz/(nrows*ncols) 
    << std::endl;
}

