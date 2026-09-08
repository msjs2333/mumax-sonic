# P1 blind listening check

The CLI tests synthetic/demo labels only. It provides 20 balanced left/right
trials and 20 balanced same-position positive/negative trials. No result is
created by default:

```text
python scripts/listening_check.py
```

To opt into actual OpenAL playback, set a comfortable headphone volume and run:

```text
python scripts/listening_check.py --play --calibrate --headphones "model" --system-volume-note "Windows 20%"
```

Each run writes a new CSV below `local/listening_trials`; existing results are
never overwritten. The CSV records device metadata, HRTF setting, volume,
response and response time (from the prompt after the sound ends, not onset reaction time).
HRTF and endpoint are read from the engine; system volume is an uncalibrated note.
Each trial sounds for one second, and calibration provides labelled examples.
Accuracy is reported separately for the two tasks. The 90% threshold is an initial engineering target
for this small check, not a scientific conclusion. Synthetic labels do not
validate magnetic topology or q/Q calculations.
