set -xe

INSTANCE="$1"

ssh $INSTANCE "sudo apt-get -y install rsync python3-venv gcc python3-dev tmux less g++ make patch"
ssh $INSTANCE "mkdir -p deepsat/"

bash remote/rsync.sh $INSTANCE

scp remote/setup.sh "$INSTANCE:"

ssh $INSTANCE "python3 -m venv deepsat/venv && source setup.sh && pip install -U pip setuptools && pip install -r deepsat/deepsat/requirements.txt"
