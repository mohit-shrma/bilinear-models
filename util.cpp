#include "util.h"


unordered_set<int> getPosUsers(gk_csr_t *mat) {
  unordered_set<int> userSet;
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

unordered_set<int> getItemSet(gk_csr_t *mat) {
  
  unordered_set<int> itemSet;
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      itemSet.insert(mat->rowind[ii]);
    }
  }
  
  return itemSet;
}


