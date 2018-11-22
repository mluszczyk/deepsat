INSTANCE="$1"

function rsync_project {
    rsync -rvx --exclude .git --exclude __pycache__ --exclude summaries . $INSTANCE:deepsat/deepsat/
}

function setup_instance {
    ssh $INSTANCE -- "sudo apt-get -y install rsync python3-venv gcc python3-dev tmux less g++ make patch zlib1g-dev"
    ssh $INSTANCE -- "mkdir -p deepsat/"

    rsync_project

    scp remote/setup.sh "$INSTANCE:"

    ssh $INSTANCE -- "python3 -m venv deepsat/venv && source setup.sh && pip install -U pip setuptools && pip install -r deepsat/deepsat/requirements.txt"
}

function create_instance {
    NAME="$1"
    gcloud compute instances create "$1" \
        --image-family debian-9 \
        --image-project debian-cloud \
        --machine-type=n1-standard-16
}

function remote_run {
    rsync_project
    ssh $INSTANCE -- "source setup.sh && cd deepsat/deepsat && cat /dev/null >stdout 2>stderr && nohup python run_experiments.py > stdout 2>stderr & tail -F ~/deepsat/deepsat/stdout ~/deepsat/deepsat/stderr"
}

function setup_env_for_gcloud {
    # Redefine ssh, scp and rsync. Note: it's better than setting a variable for SSH, RSYNC etc.
    # https://stackoverflow.com/questions/15468689/missing-trailing-in-remote-shell-command
    function ssh {
        gcloud compute ssh "$@"
    }
    function scp {
        gcloud compute scp "$@"
    }
    function rsync {
        /usr/bin/rsync -e ./remote/gcloud_ssh_replacement.sh "$@"
    }
}

function create_and_run {
    create_instance "$INSTANCE"
    setup_env_for_gcloud
    setup_instance
    remote_run
}
