function train {
	DIFFICULTY="$1"
	TEMPLATE="$2"
	LEVEL_NUMBER="$3"
	ATTENTION="$4"
	RELU_ATTENTION="$6"
	TPU_NAME="ng-tpu-$5"
	BUCKET_NAME="neural-guidance-tensorflow"
	MODEL_NAME="0409_relu_attention_henryk/${DIFFICULTY}_l${LEVEL_NUMBER}_a${ATTENTION}_ra${RELU_ATTENTION}_t${TPU_NAME}"

	SESSION="$MODEL_NAME"
	tmux new-s -d -s "$SESSION"
	tmux send-keys -t "$SESSION:0" "cd deepsat" Enter
	tmux send-keys -t "$SESSION:0" "source remote/setup.sh" Enter
	tmux send-keys -t "$SESSION:0" "MODEL_NAME=\"$MODEL_NAME\"" Enter
	tmux send-keys -t "$SESSION:0" "TPU_NAME=\"$TPU_NAME\"" Enter
	tmux send-keys -t "$SESSION:0" "ATTENTION=\"$ATTENTION\"" Enter
	tmux send-keys -t "$SESSION:0" "LEVEL_NUMBER=\"$LEVEL_NUMBER\"" Enter
	tmux send-keys -t "$SESSION:0" "BUCKET_NAME=\"$BUCKET_NAME\"" Enter
	tmux send-keys -t "$SESSION:0" "RELU_ATTENTION=\"$RELU_ATTENTION\"" Enter
	tmux send-keys -t "$SESSION:0" "$TEMPLATE" Enter
}

function sr30 {
	TEMPLATE='python neurosat_tpu.py --use_tpu=True --tpu=$TPU_NAME --train_file=gs://$BUCKET_NAME/sr30_75x1e4_uncom_train/train_2*.tfrecord --test_file=gs://$BUCKET_NAME/sr30_10x1e4_uncom_test/* --train_steps=1200000 --test_steps=80 --model_dir=gs://$BUCKET_NAME/$MODEL_NAME --export_dir=gs://$BUCKET_NAME/export/$MODEL_NAME --variable_number=30 --clause_number=300 --train_files_gzipped=False --batch_size=128 --export_model --attention=$ATTENTION --relu_attention=$RELU_ATTENTION --level_number=$LEVEL_NUMBER'
	train 30 "$TEMPLATE" "$1" "$2" "$3" "$4"
}

function sr50 {
	TEMPLATE='python neurosat_tpu.py --use_tpu=True --tpu=$TPU_NAME --train_file=gs://$BUCKET_NAME/sr_50_direct_from_prom/sr_50/train_1_sr_50.tfrecord.gz --test_file=gs://$BUCKET_NAME/sr_50_direct_from_prom/sr_50/train_2_sr_50.tfrecord.gz --train_steps=600000 --test_steps=1000 --model_dir=gs://$BUCKET_NAME/$MODEL_NAME --export_dir=gs://$BUCKET_NAME/export/$MODEL_NAME --variable_number=50 --clause_number=500 --train_files_gzipped=True --test_files_gzipped=True --batch_size=64 --export_model --attention=$ATTENTION --relu_attention=$RELU_ATTENTION --level_number=$LEVEL_NUMBER'
	train 50 "$TEMPLATE" "$1" "$2" "$3" "$4"
}

function sr70 {
	TEMPLATE='python neurosat_tpu.py --use_tpu=True --tpu=$TPU_NAME --train_file=gs://$BUCKET_NAME/sr_70_gz/train/train_1_sr_70.tfrecord.gz --test_file=gs://$BUCKET_NAME/sr_70_gz/test/test_9_sr_70.tfrecord.gz --train_steps=600000 --test_steps=1000 --model_dir=gs://$BUCKET_NAME/$MODEL_NAME --export_dir=gs://$BUCKET_NAME/export/$MODEL_NAME --variable_number=70 --clause_number=700 --train_files_gzipped=True --test_files_gzipped=True --batch_size=64 --attention=$ATTENTION --relu_attention=$RELU_ATTENTION --export_model --level_number=$LEVEL_NUMBER'
	train 70 "$TEMPLATE" "$1" "$2" "$3" "$4"
}

function sr100 {
	TEMPLATE='python neurosat_tpu.py --use_tpu=True --tpu=$TPU_NAME --train_file=gs://$BUCKET_NAME/sr_100_gz/train/train_1_sr_100.tfrecord.gz --test_file=gs://$BUCKET_NAME/sr_100_gz/test/test_9_sr_100.tfrecord.gz --train_steps=1200000 --test_steps=1000 --model_dir=gs://$BUCKET_NAME/$MODEL_NAME --export_dir=gs://$BUCKET_NAME/export/$MODEL_NAME --variable_number=100 --clause_number=1000 --train_files_gzipped=True --test_files_gzipped=True --batch_size=32 --attention=$ATTENTION --relu_attention=$RELU_ATTENTION --level_number=$LEVEL_NUMBER --export_model'
	train 100 "$TEMPLATE" "$1" "$2" "$3" "$4"
}

sr30 30 False 00 True
sr30 30 True 10 False
sr50 30 False 01 True
sr50 30 True 11 False
sr50 40 False 02 True
sr50 40 True 12 False
sr30 20 True 03 False
sr30 20 False 13 True
sr50 20 True 04 False
sr50 20 False 14 True
sr50 50 False 05 True
sr50 50 True 15 False
sr70 20 False 06 True
sr70 20 True 16 False
sr70 40 False 07 True
sr70 40 True 17 False
sr70 50 False 08 True
sr70 50 True 18 False
sr100 50 False 09 True
sr100 50 True 19 False
