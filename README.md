# Install

```
uv sync
source .venv/bin/activate # I've been using this for convenience with jupyter notebooks and torchrun, may not be necessary
huggingface-cli login --token {my token}
```


Training: `torchrun --nproc_per_node={NUM_GPUS} nl_probes/sft.py`