# Raw Experiment Reruns

The default release path reproduces paper tables and figures from the small result artifacts in `artifacts/results/`.

Raw reruns require wildfire inputs, backbone feature caches, checkpoints, and a Slurm cluster. These data paths are intentionally not hard-coded in the public repository. Set them through environment variables before submitting jobs:

```bash
export WILDFIRE_DATA_ROOT=/path/to/wildfire_inputs
export WILDFIRE_FEATURE_ROOT=/path/to/backbone_features
export WILDFIRE_OUTPUT_ROOT=/path/to/new_results
```

Use the templates in `experiments/slurm/` as the starting point. They are written so the same scripts can run on a login-safe Slurm workflow rather than executing heavy training on a login node.
