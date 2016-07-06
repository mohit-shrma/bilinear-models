import sys
import random


def getItems(ratMat):
  items = set([])
  nUsers = 0
  with open(ratMat, 'r') as f:
    for line in f:
      cols = line.strip().split()
      for i in range(0, len(cols), 2):
        items.add(int(cols[i]))
      nUsers += 1
  print 'No. of items: ', len(items)
  print 'No. of users: ', nUsers
  return list(items)


def writeMat(ratMat, items, matName, ratThresh):
  with open(ratMat, 'r') as f, open(matName, 'w') as g:
    for line in f:
      cols = line.strip().split()
      for i in range(0, len(cols), 2):
        item = int(cols[i])
        rating = float(cols[i+1])
        if item in items:
          if rating >= ratThresh:
            g.write(str(item) + ' 1 ')
          else:
            g.write(str(item) + ' 0 ')
      g.write('\n')


def main():
  ratMat    = sys.argv[1]
  seed      = int(sys.argv[2])
  ratThresh = float(sys.argv[3])

  random.seed(seed)
  
  items = getItems(ratMat)
  random.shuffle(items) 
  
  trainItems = set(items[:-2000])
  testItems  = set(items[-2000:-1000])
  valItems   = set(items[-1000:])

  writeMat(ratMat, trainItems, 'train.csr', ratThresh)  
  writeMat(ratMat, testItems, 'test.csr', ratThresh)
  writeMat(ratMat, valItems, 'val.csr', ratThresh)

if __name__ == '__main__':
  main()

