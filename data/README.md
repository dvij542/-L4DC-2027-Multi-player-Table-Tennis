# Ablation data

`ablation_training_size.csv` transcribes all nine percentages exactly from the user-supplied training-size table (November 10, 2026). The table does not specify the training-size unit or how the average static minimum is aggregated, so those details are not inferred.

`ablation_discount_factor.csv` approximately digitizes marker centers from the user-supplied plot titled `nash_q (group/gamma-associated results)`. Random-opponent values are approximately 70, 75, 75, 72.5, and 70%; static means are approximately 78, 76.5, 77.5, 75, and 76%. These are image-derived estimates, not original numerical logs. The source contains no point at gamma=0.7; the connecting line does not represent an additional measurement. No uncertainty data were supplied for either ablation.

The discount sweep is attributed to Nash-Q, as confirmed by the user. The training-size image has no method label; its caption therefore does not assign a method. The gamma=1 endpoint is an undiscounted episodic comparison; the discounted infinite-horizon tail bound in the paper applies only for gamma<1.

Regenerate both vector PDFs and PNG previews from the repository root:

```sh
python scripts/plot_ablations.py
```

Requires Python, NumPy, and Matplotlib. Plots use only these CSV values and do not fit or smooth the measured points.
