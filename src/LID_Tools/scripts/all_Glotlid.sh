for j in 0 1 2 3
do
	export SEED=$j

	nohup python3 ../GlotLID_Tools.py > ./${SEED}_chunks_out.txt \
	--seed $SEED 
done
