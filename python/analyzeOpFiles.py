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
  valRecallDic = {}
  testRecallDic = {}

  ds.append(valRecallDic)
  ds.append(testRecallDic)

  keys = set([]) 
  
  with open(fileList, 'r') as f:
    for line in f:
      fName = line.strip()
      if not os.path.isfile(fName):
        print 'NOT_FOUND: ' + fName
      else:
        isFinished = False
        with open(fName, 'r') as f:
          bName = os.path.basename(fName)
          bk = bName.strip('.txt').split('_')
          bk = ' '.join(bk)

          for line in f:
            lowLine = line.lower()
            if 'nan' in lowLine:
              print 'NAN_FOUND: ' + fName
              break

            if line.startswith('RE'):
              cols       = line.strip().split()
              valRecall  = float(cols[8])
              testRecall = float(cols[9])
              keys.add(bk)      
              updateDic(valRecallDic, bk, valRecall)
              updateDic(testRecallDic, bk, testRecall)
              isFinished = True
        if not isFinished:
          print 'NOT_COMP: ' + fName
  
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
    tempL += [valRecallDic[k][0], testRecallDic[k][0]]
    tempL += [testRecallDic[k][1]]
    print ' '.join(map(str, tempL))
              

def main():
  opListF = sys.argv[1]
  analyzeFiles(opListF)  

if __name__ == '__main__':
  main()

