#!/bin/bash

DATA="/home/grad02/mohit/exmoh/1000ItemSplits//split1"
TRAIN_FILE=$DATA"/train_user_book.mat"
TEST_FILE=$DATA"/test_user_book.mat"
VAL_FILE=$DATA"/val_user_book.mat"
FEAT_ACCUM_FILE=$DATA"/train_user_featAccu.mat"
FEAT_FILE="/home/grad02/mohit/exmoh/1000ItemSplits//"
BILINEAR="/home/grad02/mohit/exmoh/bilinearnuke/biUV"

l2regs=("0.001" "0.01" "0.1" "1")
l1regs=(0)
nucRegs=(0)
wRegs=("0.001" "0.01" "0.1" "1")
ranks=(1 5 10 20 30 50)
learnRates=("0.00001" "0.0001")

opDir=$DATA"/biUV"
mkdir $opDir

for rank in "${ranks[@]}";do
  for l2Reg in "${l2regs[@]}";do
    for l1Reg in "${l1regs[@]}";do
      for wReg in "${wRegs[@]}";do
        for nucReg in "${nucRegs[@]}";do
          for learnRate in "${learnRates[@]}";do
            echo $BILINEAR $TRAIN_FILE $TEST_FILE $VAL_FILE $FEAT_FILE \
              $FEAT_ACCUM_FILE $l1Reg \
              $l2Reg $wReg $wReg $nucReg $learnRate $rank 1 1.0 1000 1 " > " \
              $opDir"/bpr_"$l2Reg"_"$wReg"_"$learnRate"_"$rank".txt"
          done
        done
      done
    done
  done
done


