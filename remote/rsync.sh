INSTANCE="$1"
rsync -rvx --exclude .git --exclude __pycache__ --exclude summaries . $INSTANCE:deepsat/deepsat/
