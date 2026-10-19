# DL4GNSS-Benchmark

A reproducible benchmark for classical and deep-learning methods on daily GNSS
coordinate time series: denoising, gap filling, offset detection, and the
downstream product -- station velocity and its uncertainty.

The benchmark is built on 333 daily coordinate series from 321 sites (NGL and
JPL), with a station-level split, frozen station lists, fixed seeds, and
injected white-plus-flicker noise. The paper's verdict is task-specific: learned
gap filling is competitive at intermediate synthetic gaps but no better than
interpolation on the full real set, and no learned denoiser improves the
downstream velocity or its calibration. See the manuscript for the full result.

## Contents

```
experiments/
  common/                     shared numerics: GNSS trajectory models, robust LS,
                              coloured-noise MLE, learned nets, benchmark loop
  exp-20261008-data-prep/     build the frozen dataset (NGL download + JPL input)
  exp-20261008-synth-harness/ synthetic white+flicker harness
  exp-20261009-detector-fair/ offset-detection benchmark
  exp-20261009-dl-seeds/      learned denoisers/imputers, seed sweep
  exp-20261009-in-distribution/
  exp-20261009-t2-signal/     signal-recovery study
  exp-20261018-external-eval/ held-out real-series evaluation
  exp-20261018-nonparam-signals/
  exp-20261018-real-offsets/  real catalogue offset labels
  exp-20261018-rw-downstream/ random-walk downstream study
  exp-20261018-station-noise/ per-station noise study
  exp-20261018-tcn/           TCN baseline
submission/v2/                manuscript source, figure scripts, numbers ledger
out/                          compiled manuscript (v2.0)
```

Every script that produces a number or figure in the manuscript is included;
the release manifest with per-file SHA-256 is `submission/v2/release/MANIFEST.md`.

## Data

Raw coordinate series are public but are not redistributed here (size and
licensing). To rebuild the frozen dataset:

1. Download the NGL holdings catalogue:
   `https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt`
   and save it as `experiments/exp-20261008-data-prep/data/ngl_holdings.txt`.
2. The data-prep script downloads the frozen station list itself
   (`experiments/exp-20261008-data-prep/ngl_targets.txt`) from
   `https://geodesy.unr.edu/gps_timeseries/IGS14/tenv3/IGS14/<SITE>.tenv3`.
3. The JPL single-analysis-centre input is not redistributed. Point
   `jpl_source_npz` in `experiments/exp-20261008-data-prep/config.json` at your
   local copy of `clean_jpl.npz` (same schema: `sites`, `t__<site>`,
   `enu__<site>` in metres).
4. Build:

   ```bash
   cd experiments/exp-20261008-data-prep
   python run.py
   ```

   Outputs `data/series.npz`, `data/split.json`, `data/stations.csv`, and
   `results.json` (the first three are gitignored).

## Environment

Tested with Python 3.10 and:

```
numpy==2.2.6
scipy==1.15.3
pandas==2.3.3
matplotlib==3.10.9
torch==2.7.1
```

Install with `pip install -r requirements.txt`. A CUDA build of PyTorch is
optional; the learned baselines also run on CPU (slower).

## Running an experiment

Each experiment directory is self-contained and reads its `config.json`; run the
entry point from inside that directory, for example:

```bash
cd experiments/exp-20261009-t2-signal
python run.py
```

Some late-stage experiments are orchestrated by `train_tcn.py`,
`train_all.py`, `run_jobs.sh`, or `merge_partials.py` in their own directory.
Outputs are written next to the script (`results.json`, `results.csv`, figures).

## Reproducibility notes

- The station-level split is deterministic (`split.json`, seed in the config).
- The frozen NGL list is `ngl_targets.txt`; do not regenerate it unless you
  intend to change the benchmark population.
- `submission/v2/state/numbers.json` is the numeric ledger used by the paper.
- The manuscript source compiles with the standard LaTeX toolchain; see
  `submission/v2/manuscript_v2.tex`.

## Citation

If you use this benchmark, please cite the accompanying manuscript:

> Qinghua Zhang, Zhengsheng Chen. *A reproducible benchmark for deep learning on
> GNSS daily coordinate series.* Manuscript, 2026.

The repository is maintained at
<https://github.com/czsgeo/DL4GNSS-Benchmark>.

## License

No license is granted for redistribution of the manuscript or the benchmark
code beyond viewing on GitHub; contact the authors for reuse.
