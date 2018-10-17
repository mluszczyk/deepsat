set -xe

INSTANCE="$1"

bash remote/rsync.sh $INSTANCE
ssh $INSTANCE "source setup.sh && cd deepsat/deepsat && nohup python run_experiments.py > stdout 2>stderr & tail -f ~/deepsat/deepsat/stdout"
