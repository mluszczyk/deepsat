# Scripts for training on Google Cloud machines.

Write a `setup.sh` file with Neptune passwords first. You can base on `setup_example.sh`.

Run all of them with bash from the main directory and provide machine name (SSH-compatible) as the first and only command line argument. E.g.

```
bash remote/setup_instance.sh lstm-cpu-07
bash remote/remote_run.sh lstm-cpu-07
```
