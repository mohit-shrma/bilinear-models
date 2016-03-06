#ifndef _UTIL_H_
#define _UTIL_H_

#include <vector>
#include <unordered_set>
#include <Eigen/Dense>
#include "GKlib.h"

std::unordered_set<int> getItemSet(gk_csr_t *mat);
std::unordered_set<int> getPosUsers(gk_csr_t *mat);
int getNNZ(gk_csr_t *mat);
void extractFeat(gk_csr_t *itemFeatMat, int item, Eigen::VectorXf& fVec);
std::vector<std::tuple<int, int, float>> getUIRatings(gk_csr_t* mat);
#endif

