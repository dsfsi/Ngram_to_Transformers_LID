for j in 0 1 2 3
do
	export SEED=$j

	nohup python3 ../CLD_v3_Tool.py > ./${SEED}_chunks_out.txt \
	--seed $SEED 
done
