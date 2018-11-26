# Scripts for training on Google Cloud machines.

## Create and run

### Prerequisites:

- Download and configure `gcloud`.
- Run python -m `reports.spreadsheet` to acquire token.json (this is a bug).
- Write a `setup.sh` file with Neptune password. You can base on `setup_example.sh`.


### Run

```
bash remote/create_and_run.sh NEW_COMPUTE_INSTANCE_NAME
```

## More granular control

Parts of `create_and_run.sh` are available as separate scripts.

Run them with bash from the main directory and provide machine name (SSH-compatible) as the first and only command line argument. E.g.

```
bash remote/setup_instance.sh lstm-cpu-07
bash remote/remote_run.sh lstm-cpu-07
```
