#include "mathUtil.h"

float sigmoid(float x) {
  return 1.0/(1.0 + exp(-x));
}


/*
 * Try to solve nuclear-norm regularization problem:
 * arg min<X> {0.5 ||X-W||_F^2 + gamma*||X||_*}
 */
void performNucNormProj(Eigen::MatrixXf& W, float gamma) {
    
  //compute thin svd
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(W, 
      Eigen::ComputeThinU|Eigen::ComputeThinV);
 
  auto thinU = svd.matrixU();
  auto thinV = svd.matrixV();
  auto singVec = svd.singularValues();
  int zeroedCount = 0;

  //zeroed out singular values < gamma
  for (int i = 0; i < singVec.size(); i++) {
    if (singVec[i] < gamma) {
      singVec[i] = 0;
      zeroedCount++;
    } else {
      singVec[i] = singVec[i] - gamma;
    }
  }

  //update W = U*S*V^T
  W = thinU*singVec.asDiagonal()*thinV.transpose();

  std::cout << "\nZeroed count: " << zeroedCount;
}


//perform nuclear norm projection using svdlib
void performNucNormProjSVDLib(Eigen::MatrixXf& W, int rank) {
  
  int nrows = W.rows();
  int ncols = W.cols();

  //create empty dense matrix 
  DMat dW = svdNewDMat(nrows, ncols);
  //assign value in row major order
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      dW->value[i][j] = W(i,j);
    }
  }

  //convert DMat to sparse
  SMat sW = svdConvertDtoS(dW);

  //compute top-rank svd
  SVDRec svd = svdLAS2A(sW, rank);

  std::cout << "\nDimensionality: " << svd->d;
  
  //multiply singular values with Vt
  for (int i = 0; i < rank; i++) {
    for (int j = 0; j < ncols; j++) {
      svd->Vt->value[i][j] *= svd->S[i];  
    }
  }
  
  //TODO: verify
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      W(i,j) = 0; 
      for (int k = 0; k < rank; k++) {
        W(i,j) += svd->Ut->value[k][i]*svd->Vt->value[k][j]; 
      }
    }
  }



  //free 
  svdFreeDMat(dW);
  svdFreeSMat(sW);
  svdFreeSVDRec(svd);
}


float vecSpVecDot(Eigen::VectorXf& vec, gk_csr_t *mat, int row) {
  float dotp = 0;
  int colInd;
  for (int ii = mat->rowptr[row]; ii < mat->rowptr[row+1]; ii++) {
    colInd = mat->rowind[ii];
    dotp += mat->rowval[ii]*vec[colInd];
  }
  return dotp;
}

//TODO: verify correctness
void matSpVecPdt(Eigen::MatrixXf& W, gk_csr_t *mat, int row, 
    Eigen::VectorXf& pdt) {
  
  int colInd;
  float val;
  
  if (mat->ncols != W.cols()) {
    std::cerr << "\ndimensions dont match" << std::endl;
  }

  pdt.fill(0);
  //parse sparse vector
  for (int ii = mat->rowptr[row]; ii < mat->rowptr[row+1]; ii++) {
    colInd = mat->rowind[ii];
    val = mat->rowval[ii];
    //TODO: verify below
    for (int k = 0; k < W.rows(); k++) {
      pdt[k] += W(k, colInd)*val;
    } 
  }

}


//TODO: verify
void spVecMatPdt(Eigen::MatrixXf& W, gk_csr_t *mat, int row, 
    Eigen::VectorXf& pdt) {
  
  int colInd;
  float val;
  
  if (mat->ncols != W.rows()) {
    std::cerr << "\ndimensions dont match" << std::endl;
  }

  pdt.fill(0);
  //parse sparse vector
  for (int k = 0; k < W.rows(); k++) {
    for (int ii = mat->rowptr[row]; ii < mat->rowptr[row+1]; ii++) {
      colInd = mat->rowind[ii];
      val = mat->rowval[ii];
      //TODO: verify below
        pdt[k] += W(colInd, k)*val;
    }
  }

}

