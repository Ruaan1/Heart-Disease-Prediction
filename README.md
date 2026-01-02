# Heart Disease Prediction Using Probabilistic Graphical Models

This project implements a **Bayesian Network** to model and predict the presence of heart disease based on patient attributes and clinical measurements.  
The emphasis is on **probabilistic reasoning**, conditional independence assumptions, and interpretability rather than purely predictive performance.

The dataset is derived from the **Cleveland Heart Disease dataset**, a commonly used benchmark in medical ML research.

---

## 🧠 Motivation

Medical diagnosis inherently involves uncertainty. Rather than treating heart disease prediction as a black-box classification problem, this project explores a **probabilistic approach** that explicitly models relationships between variables and allows reasoning under uncertainty.

Probabilistic Graphical Models (PGMs) provide:
- interpretable structure
- principled handling of uncertainty
- transparent assumptions about conditional dependence

---

## 📁 Repository Structure

data/ → Heart disease dataset (CSV)

code/ → Preprocessing, Bayesian Network construction, and inference

report/ → Final project report (IEEE format)

---

## 🧹 Data Preprocessing

### Handling Missing Values
- Missing values were identified primarily in the `ca` and `thal` columns.
- Instead of mean or mode imputation, rows containing missing values were **removed** to avoid introducing artificial signal.
- This resulted in **297 complete records**, which remains sufficient for probabilistic modelling.

This decision prioritises **data integrity** over dataset size.

---

### Discretisation of Continuous Variables

To ensure compatibility with Bayesian Networks (which operate on categorical variables), several continuous features were discretised using clinically meaningful thresholds:

- **Age (`age_cat`)**
  - 0–40 → young  
  - 41–60 → middle  
  - 61+ → old  

- **Cholesterol (`chol_cat`)**
  - ≤200 → low  
  - 201–239 → normal  
  - ≥240 → high  

- **Resting Blood Pressure (`bp_cat`)**
  - ≤120 → low  
  - 121–139 → normal  
  - ≥140 → high  

- **Maximum Heart Rate (`thalach_cat`)**
  - ≤100 → low  
  - 101–140 → normal  
  - ≥141 → high  

- **ST Depression (`oldpeak_cat`)**
  - 0 → none  
  - 0.1–2.0 → moderate  
  - >2.0 → high  

Both raw numerical values and discretised features were retained to allow flexibility in modelling.

---

## 🔗 Model Structure

- A **Discrete Bayesian Network** was implemented using `pgmpy`.
- The target variable is `target` (presence of heart disease).
- Predictor variables:
  - `age_cat`
  - `chol_cat`
  - `bp_cat`
  - `thalach_cat`
  - `oldpeak_cat`

The network follows a **Naïve Bayes–style structure**, where each feature directly influences the target.  
This corresponds to a **conditional independence assumption** given the disease state, enabling tractable inference while remaining interpretable.

---

## 📊 Parameter Learning & Inference

- Parameters were learned using `BayesianEstimator` with a **BDeu prior** to handle sparsity and avoid zero-probability issues.
- Conditional Probability Distributions (CPDs) were extracted and inspected to validate learned relationships.
- Inference enables probabilistic queries such as:
  > *What is the probability of heart disease given high cholesterol and low maximum heart rate?*

---

## 🔍 Observations & Insights

- The model captures intuitive medical patterns:
  - Higher cholesterol and lower maximum heart rate correspond to increased disease probability.
  - Middle-aged patients form the largest demographic group.
- CPDs provide interpretable evidence of how individual risk factors contribute probabilistically rather than deterministically.

This highlights the strength of PGMs in **explainable medical decision support**.

---

## 🧠 Key Takeaways

- Probabilistic models offer transparency that black-box classifiers lack
- Explicit modelling assumptions matter in high-stakes domains like healthcare
- Discretisation choices significantly affect model behaviour and must be justified
- Bayesian Networks are well-suited for reasoning under uncertainty, even with limited data

---

## 📌 Notes

This project focuses on **interpretability and probabilistic reasoning**, not maximising predictive accuracy.  
The goal is to demonstrate principled modelling decisions rather than end-to-end performance optimisation.
