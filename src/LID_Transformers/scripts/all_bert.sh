for j in 1 2 3 4 5
do
	export MAX_LENGTH=200
	export BERT_MODEL=bert-base-uncased
	export OUTPUT_DIR=../../../../../../ext_data/neo/CTEXT_LID/bert_lid
	export TEXT_RESULT=test_result$j.txt
	export TEXT_PREDICTION=test_predictions$j.txt
	export BATCH_SIZE=128
	export NUM_EPOCHS=30
	export SAVE_STEPS=10000
	export SEED=$j

	python3 ../LID_BERT.py --data_dir ../Csv_Train_test_split/ \
	--model_type bert \
	--model_name_or_path $BERT_MODEL \
	--output_dir $OUTPUT_DIR \
	--test_result_file $TEXT_RESULT \
	--test_prediction_file $TEXT_PREDICTION \
	--max_seq_length  $MAX_LENGTH \
	--num_train_epochs $NUM_EPOCHS \
	--per_gpu_train_batch_size $BATCH_SIZE \
	--save_steps $SAVE_STEPS \
	--gradient_accumulation_steps 2 \
	--seed $SEED \
	--do_train \
	--do_eval \
	--do_predict \
	--overwrite_output_dir \
	--lang mzanzi 
done

