# Noisy Labels with Leaky ReLUs

This repository contains experimental code and results for studying the effect of **noisy labels** on **overparameterized neural networks with Leaky ReLU activations**.

---

## Repository Structure

experiment.py  
Main experiment script

checkpoint.json  
Checkpoint file for resuming experiments

results.json  
Aggregated experimental results

heatmap.png  
Heatmap of optimal clean test loss over noise rate and α

comparison.png  
Comparison between α = 0 (ReLU) and α = −1 (absolute activation)

difference.png  
Difference plot: R(α = 0) − R(α = −1)

---

## File Descriptions

### experiment.py
Runs the full experiment:
- data loading
- symmetric label noise injection
- training with Leaky ReLU activations
- early stopping based on clean test loss
- generation of all figures

### checkpoint.json
Intermediate results saved during long experiments to allow resuming.

### results.json
Final results containing loss curves and optimal stopping statistics for each noise rate and each α.

---

## Figures

### heatmap.png
Shows the minimum clean test loss (via early stopping) as a function of:
- noise rate ρ
- Leaky ReLU parameter α

Brighter colors indicate better generalization performance.

---

### comparison.png
Direct comparison of clean test loss between:
- α = 0 (ReLU)
- α = −1 (absolute value activation)

across different noise rates.

---

### difference.png
Displays the difference

R(T*) at α = 0 minus R(T*) at α = −1

Positive values indicate that α = −1 performs better.  
Negative values indicate that α = 0 performs better.

---

## Usage

Run the experiment with:

python experiment.py

Required packages include:
- numpy
- torch
- matplotlib

---

## License

This repository is intended for research and academic use.
