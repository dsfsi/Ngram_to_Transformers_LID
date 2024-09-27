for j in 0 1 2 3 4 5 6 7 8 9 
do
	export SEED=$j

	nohup python3 ../LID_NBC_KNN_SVM_LR.py > ./${SEED}_chunks_out.txt \
	--seed $SEED 
done
