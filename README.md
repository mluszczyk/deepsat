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

[Workplan](https://docs.google.com/presentation/d/1VpJoB0SrUreXzPu6HHja7ToTuy7CJmTub-2Mg_Qr1d0/edit?usp=sharing)

[Presentation](https://docs.google.com/document/d/1pAdKJz3fwAE5MTa77yuZKy8FCkn_H04nxdJ0lbxpDTM/edit?usp=sharing)

### References:

[Neurosat](https://arxiv.org/pdf/1802.03685.pdf, implementacja: https://github.com/dselsam/neurosat)

[Proof synthesis for propositional logic](https://arxiv.org/abs/1805.11799)

Other references from Neurosat !!!

EqNet

DeepMath 

GamePad



### Old repo (Theano dependent)


