# Hybrid Movie Recommendation System for Bengali OTT Streaming Catalogs

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://movie-ai1.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Abstract

Recommender systems in regional streaming media face pronounced cold-start challenges, extreme sparsity, and diverse linguistic preferences. This project implements an **end-to-end multi-modal recommendation platform** tailored to the South Asian Bengali entertainment streaming ecosystem (**Chorki** and **Hoichoi**). Evaluating an empirical dataset of **105,156 user ratings** across **381 films and series**, we develop, evaluate, and deploy four core algorithms: **TF-IDF Content-Based Filtering**, **User-KNN Collaborative Filtering**, **Truncated Singular Value Decomposition (Truncated SVD)**, and a **Weighted Hybrid Consensus Ensemble**. The SVD model achieves a state-of-the-art **Root Mean Squared Error (RMSE) of 0.454** with an inference latency under 2 ms. The accompanying web application integrates **Explainable AI (XAI)** attribution badges, an interactive **"Rate Your Taste"** latent vector builder, platform-exclusive catalog filters, and discovery serendipity tuning.

**Keywords:** Recommender Systems, Matrix Factorization, Truncated SVD, Collaborative Filtering, Content-Based Filtering, Explainable AI (XAI), Bengali Cinema, OTT Streaming.

---

## 🌐 Live Interactive Application
The recommendation dashboard is deployed on Streamlit Cloud:  
👉 **[https://movie-ai1.streamlit.app/](https://movie-ai1.streamlit.app/)**

---

## Table of Contents
- [1. Domain Overview & Regional Streaming Context](#1-domain-overview--regional-streaming-context)
- [2. Dataset Architecture & Characteristics](#2-dataset-architecture--characteristics)
- [3. Algorithmic Formulations & Methodology](#3-algorithmic-formulations--methodology)
  - [3.1 Content-Based TF-IDF Cosine Similarity](#31-content-based-tf-idf-cosine-similarity)
  - [3.2 User-Based K-Nearest Neighbors (KNN)](#32-user-based-k-nearest-neighbors-knn)
  - [3.3 Truncated SVD Matrix Factorization](#33-truncated-svd-matrix-factorization)
  - [3.4 Hybrid Weighted Rank Fusion](#34-hybrid-weighted-rank-fusion)
  - [3.5 Explainable AI (XAI) Attribution](#35-explainable-ai-xai-attribution)
- [4. Model Performance & Comparative Evaluation](#4-model-performance--comparative-evaluation)
- [5. Interactive Application Capabilities](#5-interactive-application-capabilities)
- [6. Project Structure](#6-project-structure)
- [7. Installation & Local Execution](#7-installation--local-execution)
- [8. Academic References & Citations](#8-academic-references--citations)

---

## 1. Domain Overview & Regional Streaming Context

Over-The-Top (OTT) platforms serving regional languages require domain-sensitive recommendation mechanics. In the Bengali film and series ecosystem:
- **Chorki (Bangladesh)** emphasizes contemporary romantic thrillers, crime series, and original productions.
- **Hoichoi (India/West Bengal)** hosts classic heritage cinema alongside modern family and detective franchises.

This project unifies both catalogs into a single collaborative and content-aware discovery engine, balancing blockbuster titles against long-tail regional indie cinema.

---

## 2. Dataset Architecture & Characteristics

The benchmark dataset consists of relational metadata and sparse interaction logs:

| Metric | Measurement | Description |
|:---|:---:|:---|
| **Total Titles** | **381** | Films & series across Chorki and Hoichoi |
| **Total Interactions** | **105,156** | User ratings on a 0.5 to 5.0 discrete star scale |
| **Unique User Profiles** | **668** | Active user profiles with dense interaction histories |
| **Matrix Density** | **41.3%** | User-item interaction density across the catalog |
| **Mean User Rating** | **3.53** | Global baseline rating across all genres |

```
Dataset Schema:
├── movies.csv:  [platform_Name, movieId, title, genres, director, starring]
└── ratings.csv: [userId, movieId, rating, timestamp]
```

---

## 3. Algorithmic Formulations & Methodology

### 3.1 Content-Based TF-IDF Cosine Similarity
For each movie $i$, a composite textual document is constructed from metadata:

$$D_i = \text{genres}_i \oplus \text{director}_i \oplus \text{starring}_i$$

Term Frequency–Inverse Document Frequency (TF-IDF) feature vectors $\mathbf{v}_i \in \mathbb{R}^d$ ($d \le 5,000$) are extracted. Pairwise affinity between seed item $i$ and candidate item $j$ is computed using the cosine kernel:

$$\text{sim}(i, j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\|_2 \|\mathbf{v}_j\|_2} = \frac{\sum_{k=1}^d v_{ik} v_{jk}}{\sqrt{\sum_{k=1}^d v_{ik}^2} \sqrt{\sum_{k=1}^d v_{jk}^2}}$$

### 3.2 User-Based K-Nearest Neighbors (KNN)
Given the sparse user-item interaction matrix $R \in \mathbb{R}^{m \times n}$, the distance between target user $u$ and neighbor $v$ is determined via cosine distance:

$$d(u, v) = 1 - \frac{\mathbf{r}_u \cdot \mathbf{r}_v}{\|\mathbf{r}_u\|_2 \|\mathbf{r}_v\|_2}$$

Predicted rating $\hat{r}_{ui}$ for item $i$ by user $u$ is computed across the $K$-nearest peers $\mathcal{N}_K(u)$:

$$\hat{r}_{ui} = \frac{\sum_{v \in \mathcal{N}_K(u)} \text{sim}(u, v) \cdot r_{vi}}{\sum_{v \in \mathcal{N}_K(u)} |\text{sim}(u, v)|}$$

### 3.3 Truncated SVD Matrix Factorization
Matrix factorization projects the sparse rating matrix $R$ into a $k$-dimensional latent feature space ($k = 50$):

$$R \approx U_k \Sigma_k V_k^T$$

The regularized objective function minimizes reconstruction error over observed ratings $\mathcal{K}$:

$$\min_{P, Q} \sum_{(u, i) \in \mathcal{K}} (r_{ui} - \mathbf{p}_u^T \mathbf{q}_i)^2 + \lambda (\|\mathbf{p}_u\|_2^2 + \|\mathbf{q}_i\|_2^2)$$

where $\mathbf{p}_u \in \mathbb{R}^k$ and $\mathbf{q}_i \in \mathbb{R}^k$ represent latent user and item factors, respectively. This low-rank projection mitigates the curse of dimensionality and captures latent genre/style associations.

### 3.4 Hybrid Weighted Rank Fusion
To balance content novelty and collaborative peer consensus, the hybrid engine fuses rankings via linear combination:

$$\text{Score}_{\text{hybrid}}(i) = w_{\text{content}} \cdot \frac{1}{\text{rank}_{\text{content}}(i)} + w_{\text{knn}} \cdot \frac{1}{\text{rank}_{\text{knn}}(i)} + w_{\text{svd}} \cdot \frac{1}{\text{rank}_{\text{svd}}(i)}$$

subject to $w_{\text{content}} + w_{\text{knn}} + w_{\text{svd}} = 1.0$.

### 3.5 Explainable AI (XAI) Attribution
Rather than providing opaque lists, the platform generates a transparent justification token for each recommendation:

$$\text{Attribution}(i) = \left( \mathcal{G}_0 \cap \mathcal{G}_i \right) \cup \mathbb{I}(\text{Director}_0 = \text{Director}_i) \cup \left( \mathcal{C}_0 \cap \mathcal{C}_i \right)$$

This yields plain-English rationales:  
> *"🎯 Match: Shared genre (Romantic Thriller) · Same director (Vicky Zahed) · Similarity: 89.4%"*

---

## 4. Model Performance & Comparative Evaluation

| Model Architecture | RMSE | Catalog Coverage | Latency (ms) | Cold-Start Resilience |
|:---|:---:|:---:|:---:|:---:|
| **Content-Based (TF-IDF)** | $1.200$ | **95.0%** | $3.2\text{ ms}$ | **High** |
| **Collaborative Filtering (KNN)** | $1.080$ | $75.2\%$ | $12.8\text{ ms}$ | Low |
| **Truncated SVD (Matrix Factorization)** | **0.454** | $86.4\%$ | **1.8 ms** | Medium |
| **Hybrid Ensemble** | $0.850$ | $91.8\%$ | $8.4\text{ ms}$ | **High** |

> **Key Takeaway:** Truncated SVD achieves the lowest error ($\text{RMSE} = 0.454$), while the Hybrid Ensemble provides the best balance between predictive accuracy and broad catalog coverage ($91.8\%$).

---

## 5. Interactive Application Capabilities

1. **"Rate Your Taste" Interactive Onboarding Simulator**:
   - Visitors rate 3–5 movies with 1–5 stars to instantly construct a synthetic taste vector $\mathbf{u}$.
   - The vector is projected into latent SVD space ($\hat{\mathbf{r}} = V_k \Sigma_k^{-1} U_k^T \mathbf{u}$) to generate real-time recommendations without requiring pre-existing user IDs.
2. **Platform Filtering**:
   - Toggle between **Chorki Only**, **Hoichoi Only**, or **All Catalogs**.
3. **Serendipity / Discovery Slider**:
   - Adjusts between proven blockbusters and long-tail hidden gems.
4. **Persistent Session State & CSV Export**:
   - Recommendations remain visible across tab transitions with 1-click forensic CSV download.

---

## 6. Project Structure

```
Movie_Recommendation_System/
├── app.py                                           # Streamlit interactive application
├── Movie_Recommendation_System.ipynb               # Full EDA and model experimentation
├── Movie_Recommendation_System__Model_Improvement.ipynb # SVD optimization & tuning
├── requirements.txt                                 # Optimized Python dependencies
├── README.md                                        # Academic research documentation
├── .gitignore                                       # Git exclusion rules
├── .github/
│   └── workflows/
│       └── keep_alive.yml                           # 24/7 Playwright keep-alive bot
└── dataset/
    ├── movies.csv                                   # 381 Bengali movies (Chorki & Hoichoi)
    └── ratings.csv                                  # 105,156 user ratings
```

---

## 7. Installation & Local Execution

### Prerequisites
- Python 3.10+
- pip package manager

```bash
# Clone the repository
git clone https://github.com/bipulhstu/Movie_Recommendation_System.git
cd Movie_Recommendation_System

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit Application
streamlit run app.py
```

---

## 8. Academic References & Citations

1. **Koren, Y., Bell, R., & Volinsky, C.** (2009). *Matrix Factorization Techniques for Recommender Systems.* Computer, 42(8), pp. 30–37. DOI: [10.1109/MC.2009.263](https://doi.org/10.1109/MC.2009.263).
2. **Sarwar, B., Karypis, G., Konstan, J., & Riedl, J.** (2001). *Item-based collaborative filtering recommendation algorithms.* Proceedings of the 10th International Conference on World Wide Web (WWW), pp. 285–295. DOI: [10.1145/371920.372071](https://doi.org/10.1145/371920.372071).
3. **Ricci, F., Rokach, L., & Shapira, B.** (2011). *Introduction to Recommender Systems Handbook.* Recommender Systems Handbook, Springer, pp. 1–35. DOI: [10.1007/978-0-387-85820-3_1](https://doi.org/10.1007/978-0-387-85820-3_1).
4. **Pedregosa, F. et al.** (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, pp. 2825–2830.

---

**🎬 Built for Regional OTT Streaming Media Research & Machine Learning Portfolios**
