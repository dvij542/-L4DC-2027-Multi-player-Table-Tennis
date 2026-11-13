# Ablation data

Both CSV files contain the alpha-potential measurements supplied by the user on November 13, 2026. These exact tabulated values supersede the earlier training-size table and digitized discount-factor image.

- `ablation_training_size.csv`: 500, 5,000, and 50,000 collected rallies. The collection-size unit is rallies.
- `ablation_discount_factor.csv`: discount factors 0.5, 0.6, 0.7, 0.8, 0.9, and 1.0. Every plotted marker corresponds to a supplied measurement.

Both files include all five supplied metrics: static mean, static minimum, success against Random, success against Q-IBR, and success against Other proposed. The static-minimum column uses the supplied label, without the earlier averaging qualifier. The identity of Other proposed is not inferred from the data. The gamma=1 endpoint represents undiscounted episodic returns.

Regenerate both vector PDFs and PNG previews from the repository root:

```sh
python scripts/plot_ablations.py
```

Requires Python, NumPy, and Matplotlib. Figures display static mean, static minimum, success against Random, and success against Q-IBR, using consistent colors and markers. The supplied Other proposed measurements remain in the CSV files and are excluded from both figures. Collection size uses a logarithmic axis; discount factor uses a linear axis. Lines connect supplied measurements without fitting or smoothing, and the four displayed metrics are included in both plots.
