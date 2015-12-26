#ifndef _UTIL_H_
#define _UTIL_H_
  
#include <unordered_set>

unordered_set<int> getItemSet(gk_csr_t *mat);
unordered_set<int> getPosUsers(gk_csr_t *mat);

#endif

