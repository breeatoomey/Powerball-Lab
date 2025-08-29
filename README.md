# Powerball Lab 🎲

**Author:** Breea Toomey  

Powerball-Lab is a data exploration and modeling project that investigates U.S. Powerball results using statistical and machine learning methods. It combines:

- **Historical frequency analysis** with recency bias and co-occurrence heatmaps  
- **Empirical weighting** for candidate generation  
- **Machine Learning models** (Logistic Regression and Gradient-Boosted Trees) trained on **calendar** and **astrological features**  
- **Interactive visualizations** (Plotly + Matplotlib) for analysis  

⚠️ **Disclaimer:** This project is for **educational and exploratory purposes only**. Powerball is a random lottery system; no method here improves the true odds of winning.

---

## 📦 Installation

```bash
git clone <this-repo>
cd powerball-lab-mlrepo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pip install -e .
```

---

## ▶️ Usage

All commands are exposed via the CLI:

```bash
python -m powerball_lab.cli <subcommand> [options]
```

### Core Subcommands

#### 1. `run-all`

Fetches history, generates frequency plots + interactive co-occurrence heatmap, and outputs top candidates via empirical weights.

```bash
python -m powerball_lab.cli run-all
```

Outputs:
- `out/freq_whites.png`  
- `out/freq_pb.png`  
- `out/heatmap_cooccurrence.html`  
- `out/powerball_candidates.csv`  

#### 2. `heatmap`

Generate interactive co-occurrence heatmaps (raw or normalized "lift").

```bash
# Raw co-occurrence counts
python -m powerball_lab.cli heatmap --outfile out/heatmap_cooccurrence.html

# Lift vs independence
python -m powerball_lab.cli heatmap --normalize --outfile out/heatmap_lift.html
```

#### 3. `candidates`

Sample tickets from empirical/recency-biased weights.

```bash
python -m powerball_lab.cli candidates --n 20 --halflife 500 --outfile out/empirical_candidates.csv
```

Parameters:
- `--n`: number of tickets (default 10)  
- `--halflife`: half-life in draws for recency weighting (default 500)  
- `--seed`: RNG seed (default 42)  

#### 4. `ml-candidates`

Train ML models over **calendar** + **astro** features and sample candidates.

```bash
# Logistic Regression (default)
python -m powerball_lab.cli ml-candidates --model logreg --n 50 --outfile out/ml_candidates_logreg.csv

# Gradient-Boosted Trees
python -m powerball_lab.cli ml-candidates --model gbdt --n 50 --outfile out/ml_candidates_gbdt.csv
```

Parameters:
- `--model`: `logreg` or `gbdt` (default `logreg`)  
- `--n`: number of tickets (default 20)  
- `--max-iter`: max iterations (for logreg)  
- `--seed`: RNG seed (default 42)  

#### 5. `ml-importances`

Export and visualize feature importances.

```bash
# Logistic Regression importances
python -m powerball_lab.cli ml-importances --model logreg --topk 12

# Gradient-Boosted Trees importances
python -m powerball_lab.cli ml-importances --model gbdt --topk 12
```

Outputs:
- CSVs + PNG bar charts in `out/`

---

## 📊 Methods & Equations

### 1. Empirical Weights + Heatmap

Each number’s weight is based on its historical frequency, optionally decayed with a half-life:

$$
w(n) = \sum_{i=1}^{T} \exp\!\left(-\lambda \cdot \text{age}_i\right) \cdot \mathbf{1}\{n \in \text{draw}_i\}
$$

where:
- \( \lambda = \frac{\ln 2}{\text{half-life}} \)  
- \( \mathbf{1}\{n \in \text{draw}_i\} \) is an indicator if number \(n\) appeared  

Tickets are sampled without replacement for whites, proportional to these weights, plus one red ball from its own weights.

The **co-occurrence heatmap** computes:

$$
C(a, b) = \sum_{i=1}^{T} \mathbf{1}\{a \in \text{draw}_i, b \in \text{draw}_i\}
$$

Normalized “lift” adjusts for independence:

$$
\text{Lift}(a,b) = \frac{C(a,b)}{\mathbb{E}[C(a,b)]}
$$

---

### 2. Logistic Regression

We train a **one-vs-rest classifier** for each number \(k\).

For a feature vector \(\mathbf{x}\) (calendar + astro):

$$
P(y_k = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}_k^\top \mathbf{x} + b_k)
$$

where:
- \( \sigma(z) = \frac{1}{1+e^{-z}} \) is the logistic function  
- \( \mathbf{w}_k \) is the coefficient vector for number \(k\)  

Final probabilities are normalized and used to sample tickets.

---

### 3. Gradient-Boosted Decision Trees (GBDT)

For each number \(k\), we train a boosted ensemble:

$$
F_k(\mathbf{x}) = \sum_{m=1}^{M} \gamma_m h_m(\mathbf{x})
$$

where:
- \( h_m \) are decision trees  
- \( \gamma_m \) are weights from gradient descent steps  

Probabilities are computed as:

$$
P(y_k = 1 \mid \mathbf{x}) = \sigma(F_k(\mathbf{x}))
$$

Feature importances come from average tree splits.

---

## 🗂️ Features Used

- **Calendar:**  
  - Day of week (`dow`)  
  - Month, Day-of-year  
  - Cyclical encodings (`sin`, `cos`)  

- **Astro:**  
  - Moon phase (radians)  
  - Sun ecliptic longitude (degrees)  
  - Moon ecliptic longitude (degrees)  
  - Mercury retrograde flag (binary)  

---

## 🧪 Example Workflow

```bash
# Run the whole pipeline
python -m powerball_lab.cli run-all

# Explore co-occurrence
open out/heatmap_cooccurrence.html

# Generate ML candidates with Logistic Regression
python -m powerball_lab.cli ml-candidates --model logreg --n 100 --outfile out/ml_logreg.csv

# Inspect feature importances
python -m powerball_lab.cli ml-importances --model gbdt --topk 15
open out/feature_importances_whites_gbdt.png
```

---

## ✨ Closing Note

This project is a personal exploration of statistical modeling, visualization, and machine learning applied to Powerball data.  
It was developed end-to-end by **Breea Toomey** as a showcase of applied AI/ML, data engineering, and visualization skills.

---
