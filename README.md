# traccc benchmark bot

Tools to benchmark traccc and post performance reports to GitHub.

## Compute benchmarks

* `benchmark.py` benchmarks and compares traccc performance commit-by-commit and produces a CSV.
* `plot.py` plots the output of `benchmark.py`.
* `parse_profile.py` takes an NSight Compute profile captured with `--section LaunchStats --section Occupancy --metrics gpu__time_duration.sum` and converts it into an easy-to-process CSV.

Supporting code lives in `traccc_bench_tools/`.

## Physics plots

* `run_seeding_example.py` runs a compiled CUDA seeding example and gathers CSV data:

  ```bash
  $ uv run python run_seeding_example.py .../build/bin/traccc_seeding_example_cuda output_directory
  ```

* `make_plots.py` produces plots from that data:

  ```bash
  $ uv run python make_plots.py -i output_directory "Current commit" plot_directory
  ```

  Pass multiple `-i` flags to make comparison plots:

  ```bash
  $ uv run python make_plots.py -i out1 "Current commit" -i out2 "Previous commit" plot_directory
  ```

Supporting code lives in `traccc_physics_plots/`.

## PR-driven checks

* `check_compute.py`, `check_physics.py`, `check_detray.py` run against a GitHub PR and post a summary comment.
* `listen.sh` reads PR IDs from a named pipe and dispatches the appropriate check.

## Run with

All tools are designed to run with uv, e.g.:

```bash
$ uv run python benchmark.py ...
```
