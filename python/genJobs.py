import sys
import os

REGS    = [0.001, 0.01, 0.1, 1, 10]

DIAG_L1_REGS = [0]
DIAG_L2_REGS = REGS
NDIAG_L1_REGS = [0]
NDIAG_L2_REGS = REGS

RANKS   = [1, 5, 10]

LEARN_RATE = 0.0001
NSPLITS = 1

def genJobs(prog, data, featMat, prefix):
  for i in range(1, NSPLITS+1):
    
    train = os.path.join(data, 'split'+str(i), 'train.csr')
    test = os.path.join(data, 'split'+str(i), 'test.csr')
    val  = os.path.join(data, 'split'+str(i), 'val.csr')
    featAccu = os.path.join(data, 'split'+str(i), 'train_user_featAccu.mat')
    opDir = os.path.join(data, 'split'+str(i), prefix)
    
    if not os.path.exists(opDir):
      os.mkdir(opDir)

    for dl1_reg in DIAG_L1_REGS:
      for dl2_reg in DIAG_L2_REGS:
        for ndl1_reg in NDIAG_L1_REGS:
          for ndl2_reg in NDIAG_L2_REGS:
            for rank in RANKS:
              jobStr = '_'.join(map(str, [dl1_reg, dl2_reg, ndl1_reg, ndl2_reg,
                rank, LEARN_RATE]))
              print prog, train, test, val, featMat, featAccu, ndl1_reg, \
                ndl2_reg, dl1_reg, dl2_reg, 0, LEARN_RATE, rank, 1, 1, 1000, \
                1, " > " + opDir + "/" + prefix + "_" + jobStr + ".txt"


def main():
  prog        = sys.argv[1]
  data        = sys.argv[2]
  featMat     = sys.argv[3]
  prefix      = sys.argv[4]
  
  genJobs(prog, data, featMat, prefix)

if __name__ == '__main__':
  main()

