#ifndef _UTIL_H_
#define _UTIL_H_
  
#include <unordered_set>
#include <Eigen/Dense>

unordered_set<int> getItemSet(gk_csr_t *mat);
unordered_set<int> getPosUsers(gk_csr_t *mat);
int getNNZ(gk_csr_t *mat);
void extractFeat(gk_csr_t *itemFeatMat, int item, Eigen::VectorXf& fVec);

#endif

