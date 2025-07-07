# 🧠 PCA & Classification of LFP Data from Motor Cortex

This MATLAB project processes Local Field Potential (LFP) signals recorded from multiple brain areas during a behavioral task. 
The goal is to **reduce dimensionality using PCA**, visualize clustering by brain area, 
and evaluate **how well brain regions can be decoded** from the neural activity using a linear classifier.

---

## 📊 Objective

For each subject and signal type (`go`, `cue`, `mov`):

1. Combine trials across brain areas
2. Normalize and smooth LFP signals
3. Apply **Principal Component Analysis (PCA)**
4. Visualize 2D/3D PCA scores
5. Use top 10 PCs to **classify brain area**
6. Plot **confusion matrix** of prediction accuracy

---

## 📂 Project Structure

.
├── analyze_pca.m                # Main script: runs PCA + classification per subject
├── preprocessing(...)       # Combines LFP data per signal type
├── run_pca(...)             # Runs PCA and triggers visualization/classification
├── edit_signal(...)         # Z-score + optional smoothing
├── scatter_plot(...)        # 2D/3D PCA visualization
├── scree_plot(...)          # Explained variance plot
├── decoder(...)             # Linear classifier + confusion matrix
├── plot_conf_chart(...)     # Confusion matrix (raw counts)
├── plot_conf_perc(...)      # Confusion matrix (percentage view)
├── get_brain_labels(...)    # Converts area index to string labels


---

## 🧪 Data Assumptions

- Data files are named like: `lfp_c.mat`, `lfp_m.mat`, `lfp_p.mat`
- Each file contains fields like: m1_go, m1_cue, m1_mov, pm_go etc

  - Each field is a matrix of shape `[trials × time]`, aligned to behavioral events.
- Brain areas analyzed: `'pm'`, `'m1'`, `'ss'`

---

## 🧮 Key Techniques

- **Z-score normalization** per trial
- **Smoothing** with adjustable window (default = 10 samples)
- **Dimensionality reduction** via `pca()`
- **Linear classification** with `fitcdiscr` and `cvpartition`
- **Confusion matrices** averaged over 1000 holdout splits

---

## 🖼️ Output Files

For each subject and signal type:

- `scatter_2_<subject>_<signal>.png` — 2D PCA projection by brain area
- `scatter_3_<subject>_<signal>.png` — 3D PCA projection
- `scree_<signal>_<subject>.fig` — scree plot (variance explained)
- `<subject>_confusion_<signal>.png` — confusion matrix (raw counts)
- `<subject>_confusionP_<signal>.png` — confusion matrix (percent view)

---

## ⚙️ Requirements

- MATLAB R2019b or newer
- Statistics and Machine Learning Toolbox (for `fitcdiscr`, `cvpartition`, etc.)

---

## 🚀 How to Run

1. Place all subject `.mat` files under the `lfp/` folder.
2. Run the main script:
 ```matlab
 analyze_pca()

Created by Almog Miron
For academic use in neural signal analysis, dimensionality reduction, and decoding.
