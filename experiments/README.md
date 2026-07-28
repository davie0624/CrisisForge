# Experiment registry

Every experiment receives a stable ID and one directory under `experiments/runs/`.
The directory records:

- frozen configuration and data-manifest hash;
- git commit;
- random seeds and environment;
- command and wall-clock time;
- metrics with uncertainty;
- artifacts and checkpoint hashes;
- decision, limitations, and next action.

Negative results are first-class records. They are never deleted merely because a
more complex model failed to beat a baseline.
