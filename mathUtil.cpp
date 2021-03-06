#include "mathUtil.h"

float proxL1(float x, float l1Reg) {
  if (fabs(x) > l1Reg) {
    if (x > 0) {
      x = fabs(x) - l1Reg;
    } else {
      x = -1.0 * (fabs(x) -  l1Reg);
    }
  } else  {
    x = 0;
  }
  return x;
}


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


//perform nuclear norm projection using svdlib with nucnorm coeff
void performNucNormProjSVDLibWReg(Eigen::MatrixXf& W, float gamma) {
  
  int nrows = W.rows();
  int ncols = W.cols();
  int rank, qrank, i;

  int minrowcol = W.rows() < W.cols() ? W.rows() : W.cols();
  rank = minrowcol < 50? minrowcol:50;

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
  std::cout << "\nNuke reg: " << gamma; 
  //find rank at which singular values dont exceed gamma
  for (i = 0; i < rank; i++) {
    if (svd->S[i] < gamma) {
      break;
    }
  }
  qrank = i;

  if (qrank == 0) {
    std::cerr << "\nperformNucNormProjSVDLibWReg: all singular values below gamma"<< std::endl;
    exit(0);
  }
  
  std::cout << "\nTrimmed rank = " << qrank << " S[" << qrank-1 <<"]: " 
    << svd->S[qrank-1] << " S[0]: " << svd->S[0] << std::endl;

  //multiply singular values with Vt
  for (i = 0; i < qrank; i++) {
    for (int j = 0; j < ncols; j++) {
      svd->Vt->value[i][j] *= svd->S[i];  
    }
  }
  
  for (i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      W(i,j) = 0; 
      for (int k = 0; k < qrank; k++) {
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


//compute W*[f_i1, f_i2 ... ]
void matSpVecsPdt(Eigen::MatrixXf& W, gk_csr_t *mat, 
    const std::vector<int>& inds, Eigen::MatrixXf& pdt) {
  
  int colInd;
  float val;
  pdt.fill(0);

  if (mat->ncols != W.cols()) {
    std::cerr << "\nmatSpVecsPdt: dimensions dont match: (" 
      << mat->nrows << "," << mat->ncols << ") (" 
      << W.rows() << "," << W.cols() << ")" <<  std::endl;
  }

  if (pdt.cols() != inds.size()) {
    std::cerr << "matSpVecsPdt: nItems and n cols in product matrix don't match" 
      << std::endl;
  }
  
  const int ncols = inds.size();

  for (int j = 0; j < ncols; j++) {
    int ind = inds[j];
    for (int ii = mat->rowptr[ind]; ii < mat->rowptr[ind+1]; ii++) {
      colInd = mat->rowind[ii];
      val = mat->rowval[ii];
      for (int k = 0; k < W.rows(); k++) {
        pdt(k, j) += W(k, colInd)*val;
      }
    }
  }

}


void wUVSpVecsPdt(Eigen::VectorXf& w, Eigen::MatrixXf& U, Eigen::MatrixXf& V, gk_csr_t *mat, 
    const std::vector<int>& inds, Eigen::MatrixXf& pdt) {
  
  int colInd;
  float val;
  
  pdt.fill(0);
  Eigen::VectorXf vec(U.cols());

  if (mat->ncols != V.rows()) {
    std::cerr << "\nUVSpVecsPdt: dimensions dont match: (" 
      << mat->nrows << "," << mat->ncols << ") (" 
      << V.rows() << "," << V.cols() << ")" <<  std::endl;
  }

  if (pdt.cols() != inds.size()) {
    std::cerr << "matSpVecsPdt: nItems and n cols in product matrix don't match" 
      << std::endl;
  }
  
  const int ncols = inds.size();

  for (int j = 0; j < ncols; j++) {
    int ind = inds[j];
    //V^T*f_i in vec
    spVecMatPdt(V, mat, ind, vec); 
    //compute U*(V^T*f_i)
    pdt.col(j) = U*vec; 
    for (int ii = mat->rowptr[ind]; ii < mat->rowptr[ind+1]; ii++) {
      colInd = mat->rowind[ii];
      val = mat->rowval[ii];
      pdt(colInd, j) += val*w(colInd);
    }
  }

}


void wSpVecsPdt(Eigen::VectorXf& w, gk_csr_t *mat, 
    const std::vector<int>& inds, Eigen::MatrixXf& pdt) {
  
  int colInd;
  float val;
  
  pdt.fill(0);

  if (pdt.cols() != inds.size()) {
    std::cerr << "matSpVecsPdt: nItems and n cols in product matrix don't match" 
      << std::endl;
  }
  
  const int ncols = inds.size();

  for (int j = 0; j < ncols; j++) {
    int ind = inds[j];
    for (int ii = mat->rowptr[ind]; ii < mat->rowptr[ind+1]; ii++) {
      colInd = mat->rowind[ii];
      val = mat->rowval[ii];
      pdt(colInd, j) += val*w(colInd);
    }
  }

}


//compute dot product of matrix and sparse vector: Ax
void matSpVecPdt(Eigen::MatrixXf& W, gk_csr_t *mat, int row, 
    Eigen::VectorXf& pdt) {
  
  int colInd;
  float val;
  
  if (mat->ncols != W.cols()) {
    std::cerr << "\nmatSpVecPdt: dimensions dont match: (" 
      << mat->nrows << "," << mat->ncols << ") (" 
      << W.rows() << "," << W.cols() << ")" <<  std::endl;
  }

  pdt.fill(0);
  //parse sparse vector
  for (int ii = mat->rowptr[row]; ii < mat->rowptr[row+1]; ii++) {
    colInd = mat->rowind[ii];
    val = mat->rowval[ii];
    for (int k = 0; k < W.rows(); k++) {
      pdt[k] += W(k, colInd)*val;
    } 
  }

}


//compute dot product of sparse vector and dense matrix: x^TA
void spVecMatPdt(Eigen::MatrixXf& W, gk_csr_t *mat, int row, 
    Eigen::VectorXf& pdt) {
  
  int colInd;
  float val;

  /*
  if (mat->ncols != W.rows()) {
    std::cerr << "\nspVecMatPdt: dimensions dont match: " << mat->ncols 
      << " " << W.rows() << " (" << mat->nrows << "," << mat->ncols <<") "
      << " (" << W.rows() << "," << W.cols() << ")" << std::endl;
    exit(0);
  }
  */

  pdt.fill(0);
  //parse sparse vector
  for (int k = 0; k < W.cols(); k++) {
    for (int ii = mat->rowptr[row]; ii < mat->rowptr[row+1]; ii++) {
      colInd = mat->rowind[ii];
      val = mat->rowval[ii];
      pdt[k] += W(colInd, k)*val;
    }
  }

}


//update matrix with sign*vec*vec^T
void updateMatWSpOuterPdt(Eigen::MatrixXf& W, gk_csr_t *mat1, int row1, 
    gk_csr_t *mat2, int row2, float scalar) {
  
  int ind1, ind2;
  int ii1, ii2;
  float val1, val2;

  for (ii2 = mat2->rowptr[row2]; ii2 < mat2->rowptr[row2+1]; ii2++) {
    ind2 = mat2->rowind[ii2];
    val2 = mat2->rowval[ii2];
    for (ii1 = mat1->rowptr[row1]; ii1 < mat1->rowptr[row1+1]; ii1++) {
      ind1 = mat1->rowind[ii1];
      val1 = mat1->rowval[ii1];
      W(ind1, ind2) += scalar*val1*val2;
    }
  }

}


void updateMatWSpOuterPdt(Eigen::MatrixXf& W, Eigen::MatrixXi& T, 
    gk_csr_t *mat1, int row1, 
    gk_csr_t *mat2, int row2, float scalar) {
  
  int ind1, ind2;
  int ii1, ii2;
  float val1, val2;

  for (ii2 = mat2->rowptr[row2]; ii2 < mat2->rowptr[row2+1]; ii2++) {
    ind2 = mat2->rowind[ii2];
    val2 = mat2->rowval[ii2];
    for (ii1 = mat1->rowptr[row1]; ii1 < mat1->rowptr[row1+1]; ii1++) {
      ind1 = mat1->rowind[ii1];
      val1 = mat1->rowval[ii1];
      W(ind1, ind2) += scalar*val1*val2;
      T(ind1, ind2) += 1;
    }
  }

}


void updateMatWSpOuterPdtWMap(Eigen::MatrixXf& W, gk_csr_t *mat1, int row1, 
    gk_csr_t *mat2, int row2, float scalar, 
    std::map<int, std::unordered_set<int>>& coords) {
  
  int ind1, ind2;
  int ii1, ii2;
  float val1, val2;
  
  for (ii2 = mat2->rowptr[row2]; ii2 < mat2->rowptr[row2+1]; ii2++) {
    ind2 = mat2->rowind[ii2];
    val2 = mat2->rowval[ii2];
    for (ii1 = mat1->rowptr[row1]; ii1 < mat1->rowptr[row1+1]; ii1++) {
      ind1 = mat1->rowind[ii1];
      val1 = mat1->rowval[ii1];
      W(ind1, ind2) += scalar*val1*val2;
      if (coords.count(ind1) == 0) {
        coords[ind1] = std::unordered_set<int>();
      }
      coords[ind1].insert(ind2);
    }
  }

}


//update matrix with sign*vec*vec^T
void lazyUpdMatWSpOuterPdt(Eigen::MatrixXf& W, Eigen::MatrixXf& T, 
    gk_csr_t *mat1, int row1, gk_csr_t *mat2, int row2, double scalar, 
    double regMult, int subIter) {
  
  int ind1, ind2;
  int ii1, ii2;
  float val1, val2;

  for (ii2 = mat2->rowptr[row2]; ii2 < mat2->rowptr[row2+1]; ii2++) {
    ind2 = mat2->rowind[ii2];
    val2 = mat2->rowval[ii2];
    for (ii1 = mat1->rowptr[row1]; ii1 < mat1->rowptr[row1+1]; ii1++) {
      ind1 = mat1->rowind[ii1];
      val1 = mat1->rowval[ii1];
      //update W(ind1, ind2)
      
      //update with reg updates
      W(ind1, ind2) = W(ind1, ind2)*pow(regMult, (subIter+1) - T(ind1, ind2));

      //record that reg update was done in this iter
      T(ind1, ind2) = subIter + 1;
      
      //actual update
      W(ind1, ind2) += scalar*val1*val2;
    }
  }

}


void lazySparseUpdMatWSpOuterPdt(Eigen::MatrixXf& W, Eigen::MatrixXf& T, 
    gk_csr_t *mat1, int row1, gk_csr_t *mat2, int row2, double scalar, 
    double regMult, int subIter, float l1Reg) {
  
  int ind1, ind2;
  int ii1, ii2;
  float val1, val2;

  for (ii2 = mat2->rowptr[row2]; ii2 < mat2->rowptr[row2+1]; ii2++) {
    ind2 = mat2->rowind[ii2];
    val2 = mat2->rowval[ii2];
    for (ii1 = mat1->rowptr[row1]; ii1 < mat1->rowptr[row1+1]; ii1++) {
      ind1 = mat1->rowind[ii1];
      val1 = mat1->rowval[ii1];
      //update W(ind1, ind2)
      
      //update with reg updates
      W(ind1, ind2) = W(ind1, ind2)*pow(regMult, (subIter+1) - T(ind1, ind2));

      //record that reg update was done in this iter
      T(ind1, ind2) = subIter + 1;
      
      //actual update
      W(ind1, ind2) += scalar*val1*val2;
      
      //L1 or proximal update
      W(ind1, ind2) = proxL1(W(ind1, ind2), l1Reg);
    }
  }

}


void lazySparseUpdMatWSpOuterPdtD(Eigen::MatrixXf& W, Eigen::MatrixXf& T, 
    gk_csr_t *mat1, int row1, gk_csr_t *mat2, int row2, double scalar, 
    double regMultDiag, double regMultNDiag, int subIter, float wl1Reg,
    float l1Reg) {
  
  int ind1, ind2;
  int ii1, ii2;
  float val1, val2;

  for (ii2 = mat2->rowptr[row2]; ii2 < mat2->rowptr[row2+1]; ii2++) {
    ind2 = mat2->rowind[ii2];
    val2 = mat2->rowval[ii2];
    for (ii1 = mat1->rowptr[row1]; ii1 < mat1->rowptr[row1+1]; ii1++) {
      ind1 = mat1->rowind[ii1];
      val1 = mat1->rowval[ii1];
      //update W(ind1, ind2)
      if (ind1 == ind2) {
        //diagonal update
        //update with reg updates
        W(ind1, ind2) = W(ind1, ind2)*pow(regMultDiag, (subIter+1) - T(ind1, ind2));

        //record that reg update was done in this iter
        T(ind1, ind2) = subIter + 1;
        
        //actual update
        W(ind1, ind2) += scalar*val1*val2;
        
        //L1 or proximal update
        W(ind1, ind2) = proxL1(W(ind1, ind2), wl1Reg);
        
      } else {
        //non-diagonal update
        //update with reg updates
        W(ind1, ind2) = W(ind1, ind2)*pow(regMultNDiag, (subIter+1) - T(ind1, ind2));

        //record that reg update was done in this iter
        T(ind1, ind2) = subIter + 1;
        
        //actual update
        W(ind1, ind2) += scalar*val1*val2;
        
        //L1 or proximal update
        W(ind1, ind2) = proxL1(W(ind1, ind2), l1Reg);
      }
    }
  }

}


//update matrix with sign*(vec1*vec2^T + vec2*vec1^T)
void updateMatWSymSpOuterPdt(Eigen::MatrixXf& W, gk_csr_t *mat1, int row1, 
    gk_csr_t *mat2, int row2, float scalar) {
  
  int ind1, ind2;
  int ii1, ii2;
  float val1, val2;

  for (ii1 = mat1->rowptr[row1]; ii1 < mat1->rowptr[row1+1]; ii1++) {
    ind1 = mat1->rowind[ii1];
    val1 = mat1->rowval[ii1];
    for (ii2 = mat2->rowptr[row2]; ii2 < mat2->rowptr[row2+1]; ii2++) {
      ind2 = mat2->rowind[ii2];
      val2 = mat2->rowval[ii2];
      W(ind1, ind2) += scalar*val1*val2;
      W(ind2, ind1) += scalar*val1*val2;
    }
  }

}


float sparseDotProd(gk_csr_t* mat1, int i, gk_csr_t* mat2, int j) {
  float sim = 0;
  for (int ii = mat1->rowptr[i]; ii < mat1->rowptr[i+1]; ii++) {
    
    int ind1   = mat1->rowind[ii];
    float val1 = mat1->rowval[ii];

    for (int ii2 = mat2->rowptr[j]; ii2 < mat2->rowptr[j+1]; ii2++) {
      
      int ind2   = mat2->rowind[ii2];
      float val2 = mat2->rowval[ii2];
      
      if (ind1 == ind2 ) {
        sim += val1*val2;
        break;
      }

    }
  } 
  return sim;
}


//NOTE: use only if columns are sorted by their indexes 
float sparseDotProd2(gk_csr_t* mat1, int i, gk_csr_t* mat2, int j) {
  float sim = 0;
  int ii1, ii2;
  int ind1, ind2;
  float val1, val2;

  ii1  = mat1->rowptr[i];
  ind1 = mat1->rowind[ii1];
  ii2  = mat2->rowptr[j];
  ind2 = mat2->rowind[ii2];
  
  while ((ii1 < mat1->rowptr[i+1]) && (ii2 < mat2->rowptr[j+1])) {
    ind1 = mat1->rowind[ii1];
    ind2 = mat2->rowind[ii2];
    if (ind1 == ind2) {
      val1 = mat1->rowval[ii1];
      val2 = mat2->rowval[ii2];
      sim += val1*val2;
      ii1++;
      ii2++;
      continue;
    }
    if (ind1 < ind2) {
      ii1++;
    } else if (ind2 < ind1) {
      ii2++;
    }    
  }

  return sim;
}


//update matrix with spVec*vec^T
void spVecVecOuterPdt(Eigen::MatrixXf& pdt, Eigen::VectorXf& vec, gk_csr_t* mat,
    int row) {
  for (int ii = mat->rowptr[row]; ii < mat->rowptr[row+1]; ii++) {
    int i = mat->rowind[ii];
    for (int j = 0; j < vec.size(); j++) {
      pdt(i,j) += mat->rowval[ii]*vec[j];
    }
  }
}


Eigen::MatrixXf spMatMatPdt(gk_csr_t *mat, Eigen::MatrixXf& W) {

  int nrows = mat->nrows;
  int ncols = W.cols();

  Eigen::MatrixXf res(nrows, ncols);
  Eigen::VectorXf pdt(ncols);
  for (int i = 0; i < nrows; i++) {
    spVecMatPdt(W, mat, i, pdt);
    res.row(i) = pdt;
  }

  return res;
}


void spVecDiff(gk_csr_t* mat1, int row1, gk_csr_t* mat2, int row2, 
    Eigen::VectorXf& res) {
  
  int ii, ind;
  float val;
  
  res.fill(0);
  for (ii = mat1->rowptr[row1]; ii < mat1->rowptr[row1+1]; ii++) {
    ind = mat1->rowind[ii];
    val = mat1->rowval[ii];
    res[ind] = val;
  }

  for (ii = mat2->rowptr[row2]; ii < mat2->rowptr[row2+1]; ii++) {
    ind = mat2->rowind[ii];
    val = mat2->rowval[ii];
    res[ind] -= val;
  }

}


//compute a.*w.*b componentwise hadamard product
float spVecWtspVecPdt(Eigen::VectorXf& w, gk_csr_t* mat1, int row1, 
    gk_csr_t* mat2, int row2) {
  float res = 0;
  for (int ii1 = mat1->rowptr[row1]; ii1 < mat1->rowptr[row1+1]; ii1++) {
    int ind1 = mat1->rowind[ii1];
    float val1 = mat1->rowval[ii1];
    for (int ii2 = mat2->rowptr[row2]; ii2 < mat2->rowptr[row2+1]; ii2++) {
      int ind2 = mat2->rowind[ii2];
      float val2 = mat2->rowval[ii2];
      if (ind1 == ind2) {
        res += w[ind1]*val1*val2;
      }
    }
  }
  return res;
}



