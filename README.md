### TPU status

Run on TPU like this:

    python neurosat_tpu.py \
        --use_tpu=True \
        --tpu=$TPU_NAME \
        --model_dir=gs://neural-guidance-tensorflow/$MODEL_NAME \
        --train_file=gs://neural-guidance-tensorflow/test-10k.tfrecord \
        --train_steps=1000
        
Set `TPU_NAME` to something like `ng-tpu-02`. Set model name to any desired unique 
name for the experiment e.g. `test-124`.
        
Tensorboard is set up to watch gs://neural-guidance-tensorflow/.

Run locally on CPU (for testing) like this:

    python neurosat_tpu.py \
        --use_tpu=False \
        --model_dir=/tmp/tb/model003 \
        --train_file=test-10k.tfrecord \
        --train_steps=100 \
        --batch_size=16


#### Generating data

Data is generated using `dump_data.py`. TPU works only with Google Storage,
so execute the following command in order to use new dataset:

    gsuitl cp test-10k.tfrecord gs://neural-guidance-tensorflow
    
    
#### Performance

With 30 var and level_number 30: ~6k steps/sec.

With 8 var and level_number 30: ~30k steps/sec.

With 8 var dataset and level_number 0:

    INFO:tensorflow:global_step/sec: 294.272
    INFO:tensorflow:examples/sec: 301335
    
    
However, with `host_call` (summaries in TensorBoard):

    INFO:tensorflow:global_step/sec: 0.874507
    INFO:tensorflow:examples/sec: 895.495

### How to run it?

#### Running on Prometheus

##### Running using srun (preferred over nothing)

In the case of Prometheus reserve a machine with bash and then run
```bash
cd /net/archive/groups/plggluna/henryk/sat_solving
source deepsat.sh
cd deepsat
```
If it is a machine without GPU then run `source deepsat_cpu.sh` instead. 

##### Using sbatch (preferred over naked srun)

TODO (jaszczur): tell us what is in the 8 sbatches and how they are different

##### Using neptune and srun (preferred, unless you are a master of grep/awk)

TODO (michal): tell how to run it with neptune

##### Using mrunner (preferred over all above)

TODO (henryk): write an mrunner script so we can run it from a laptop

#### Running on a generic machine

In the case of another machine build and activate a `python3` virtualenv using the requirements.txt file.

Once you have the basic setup run

```bash
python graph_policy.py
```
or
```bash
python neurosat_policy.py
```

#### How to run the full pipeline?

TODO: explain how to train and later test on the generic test set. Basically train and run a notebook. It would be nice to automate it and present jointly the results (in Neptune).

### Quantitive observations

#### Running time

This will generate models in the `models` subdirectory. Some rough numbers

1. On a generic i5 laptop with 4 cores ETA around 600 000 seconds.
2. On a GPU-equipeed machine on the Prometheus with 12 cores ETA around 180 000 seconds (slow).
3. On a CPU-only machine on the Promethus with 24 cores ETA around 400 000 seconds (slow).  
4. Total numbers of parameters around 132 000. 

#### Results 

TODO: add previous results, links do neptune

### Our documents:

[Workplan](https://docs.google.com/document/d/1pAdKJz3fwAE5MTa77yuZKy8FCkn_H04nxdJ0lbxpDTM/edit?usp=sharing)

[Presentation, Bumerang Workshop (Budapest), September 25, 2018](https://docs.google.com/presentation/d/1OdEJsB6DqgB0NA-eG88vmHI2uE5U0k54BRvbgXrOdwY/edit)

[Presentation, April 17, 2018](https://docs.google.com/presentation/d/1VpJoB0SrUreXzPu6HHja7ToTuy7CJmTub-2Mg_Qr1d0/edit?usp=sharing)

[Presentation, Feb 27, 2018](https://docs.google.com/presentation/d/1N0xV2XvMllsjwAKTZBAEyyKKkqdHcgAFJN4ek3sPA7A/edit)

[LSTM experiments](https://docs.google.com/document/d/1MG_PA4y0jn6vV1nTy43wvsKeK1-5uRPRFPSarLCJ2m4/edit#heading=h.nfl3p9glf50k)

### References:

Neurosat: [arxiv](https://arxiv.org/pdf/1802.03685.pdf), [code](https://github.com/dselsam/neurosat)

[Proof synthesis for propositional logic](https://arxiv.org/abs/1805.11799)

Other references from Neurosat !!!

EqNet

DeepMath 

GamePad



### Old repo (Theano dependent, related to eqnet)

https://github.com/mluszczyk/sat-notebooks



