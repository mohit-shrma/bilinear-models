import sys
import os.path


def updateDic(dic, k, val):
  if k not in dic:
    dic[k] = [0.0, 0.0]
  dic[k][0] += val
  dic[k][1] += 1


def averageDic(dic):
  for k, v in dic.iteritems():
    dic[k][0] = dic[k][0]/dic[k][1]


def analyzeFiles(fileList):
  
  ds = []
  valRecallDics = []
  testRecallDics = []

  ds.append(valRecallDics)
  ds.append(testRecallDics)

  keys = set([]) 
  
  with open(fileList, 'r') as f:
    for line in f:
      fName = line.strip()
      if not os.path.isfile(fName):
        print 'NOT_FOUND: ', fName
      else:
        with open(fName, 'r') as f:
          bName = os.path.basname(fName)
          bk = bName.strip('.txt').split('_')
          bk = ' '.join(bk)

          for line in f:
            lowLine = line.lower()
            if 'nan' in lowLine:
              print 'NAN_FOUND: ', fName
              break

            if line.startswith('RE'):
              cols       = line.strip().split()
              l2Reg      = cols[1]
              l1Reg      = cols[2]
              wl1Reg     = cols[3]
              wl2Reg     = cols[4]
              learnRate  = cols[5]
              rank       = cols[6]
              valRecall  = cols[7]
              testRecall = cols[8]
              
              updateDic(valRecallDics, bk, valRecall)
              updateDic(testRecallDics, bk, testRecall)
  
  for d in ds:
    averageDic(d)

  notFoundK = set([]) 
  for d in ds:
    for k in keys:
      if k not in d:
        print 'KEY_NOT_FOUND: ', k
        notFoundK.add(k)

  if len(notFoundK) > 0:
    print 'KEY_NOT_FOUND count: ', len(notFoundK)
    for nf in notFoundK:
      print nf
    return

  for k in keys:
    if k in notFoundK:
      continue
    tempL = [k]
    tempL += [valRecallDics[k][0], testRecallDics[k][0]]
    tempL += [testRecallDics[k][1]]
    print ' '.join(map(str, tempL))
              

def main():
  opListF = sys.argv[1]
  analyzeFiles(opListF)  

if __name__ == '__main__':
  main()

