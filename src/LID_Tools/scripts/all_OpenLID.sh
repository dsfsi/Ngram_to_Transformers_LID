for j in 0 1 2 3 5
do
	export SEED=$j

	nohup python3 ../LID_Tools_OpenLID.py > ./${SEED}_chunks_out.txt \
	--seed $SEED 
done
