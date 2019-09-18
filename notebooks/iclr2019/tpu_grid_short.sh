function train {
	DIFFICULTY="$1"
	TEMPLATE="$2"
	LEVEL_NUMBER="$3"
	ATTENTION="$4"
	RELU_ATTENTION="$6"
	TPU_NAME="ng-tpu-$5"
	BUCKET_NAME="neurosat-attention"
	MODEL_NAME="0409_attention_christian_${DIFFICULTY}_l${LEVEL_NUMBER}_a${ATTENTION}_t${TPU_NAME}"
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

sr30 30 False 00 True
