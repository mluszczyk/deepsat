#! /bin/sh
# for use with rsync
# https://stackoverflow.com/a/48105694/3576976
host="$1"
shift
exec gcloud compute ssh "$host" -- "$@"
