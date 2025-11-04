# TimeGAN for Synthetic Sequential Data Generation

## Project Overview

This repository implements **TimeGAN**, a Generative Adversarial Network (GAN) architecture designed for **time series and sequential data synthesis**.  
The model combines **autoencoding**, **supervised learning**, and **adversarial training** to create synthetic sequential data that preserve both temporal dependencies and statistical realism.

In this PyTorch implementation, TimeGAN was trained on the **Salary Data Prediction** dataset (from Kaggle), consisting of two main features — `Years` and `Salary`.  
The pipeline demonstrates the full workflow: **training, sampling, and evaluation** of the model, followed by comprehensive **visual and statistical comparisons** between real and synthetic data.

---

### 1. Architecture Overview

TimeGAN is composed of five cooperating neural components:

1. **Embedder** – Encodes real time series into a latent representation using recurrent layers.  
   This step captures temporal relationships and compresses information without losing dynamics.

2. **Recovery** – Reconstructs the original time series from the latent representation.  
   Together, the Embedder and Recovery form an **autoencoder**, ensuring that the latent space preserves meaningful temporal structure.

3. **Generator** – Produces synthetic latent sequences from random noise vectors.  
   Instead of generating directly in data space, it operates in the latent domain to ensure more stable and coherent temporal evolution.

4. **Supervisor** – Learns temporal transitions in the latent space.  
   It predicts the next latent state given the previous ones, transferring the temporal logic of real data to the Generator.

5. **Discriminator** – Distinguishes real latent sequences (from the Embedder) from fake ones (from the Generator).  
   It acts as the adversarial feedback mechanism that guides the Generator toward realism.

---

### 2. Training Philosophy

The training process of TimeGAN does not rely solely on adversarial feedback.  
Instead, it unfolds in **three coordinated phases**, each focusing on a specific learning objective:

#### **Phase 1: Autoencoder Training**
The first step focuses on training the **Embedder** and **Recovery** networks together.  
The goal is to make sure real sequences can be faithfully reconstructed from their latent representations.  
By minimizing the reconstruction error, this phase establishes a **meaningful latent space** that captures temporal dependencies.

#### **Phase 2: Supervised Training**
Next, the **Supervisor** network is trained to predict the next latent vector based on past ones.  
This phase introduces a **temporal learning signal**—essentially teaching the model how time progresses.  
It enables the Generator, later on, to create sequences that evolve realistically over time rather than random, disconnected steps.

#### **Phase 3: Joint Adversarial Training**
Finally, all networks (Generator, Supervisor, and Discriminator) are trained together in a **joint adversarial phase**.  
The Discriminator learns to differentiate real from synthetic latent trajectories, while the Generator and Supervisor cooperate to produce sequences that both **look real** and **follow real temporal transitions**.  
This adversarial feedback, combined with the supervised signal, stabilizes the training and ensures that generated data maintains **statistical fidelity** and **temporal realism**.

---

## Training Process Explanation

The training pipeline was implemented step by step to mirror the theoretical design of TimeGAN while maintaining clarity in execution.

1. **Autoencoder Phase**
   - The Embedder and Recovery were optimized together using Mean Squared Error (MSE) loss.  
   - The objective was to minimize the reconstruction gap between input data and its recovered version, establishing a reliable latent representation.

2. **Supervisor Phase**
   - The Supervisor was trained independently with a temporal MSE loss between consecutive latent states.  
   - This stage ensures that the model learns how the hidden representations evolve through time, forming a foundation for realistic sequential generation.

3. **Joint Adversarial Phase**
   - The Generator, Supervisor, and Discriminator were optimized together.  
   - The Generator aimed to fool the Discriminator, while the Discriminator learned to detect synthetic latent trajectories.  
   - The Generator’s loss included additional moment-matching penalties to align the statistical moments (mean and variance) of synthetic data with real data.  
   - During this phase, multiple updates of the Generator were performed per Discriminator step (a ratio of *k = 2*) to maintain stable adversarial learning.

Throughout the process, training stability was ensured by carefully balancing losses, using gradient clipping, and applying the Adam optimizer for all networks with consistent learning rates.

---

## Model Insights

- TimeGAN successfully blends **adversarial realism** and **temporal prediction**, allowing the Generator to learn not only what data looks like but **how it changes over time**.  
- The Supervisor component plays a crucial role, transferring temporal knowledge from real sequences to the Generator.  
- The use of an autoencoder ensures that both the structure and the sequential dependencies of the data are preserved in the latent space.  
- Once trained, the model can produce entirely new, unseen sequences that mimic real-world patterns—statistically similar but temporally coherent.

---


## Results and Evaluation

### 1. Statistical Distribution Alignment

The first set of KDE (Kernel Density Estimation) plots compares the probability density of **real** (blue) and **synthetic** (red) samples.  
The overlap (purple) illustrates that the synthetic data closely matches the real feature distributions.

![KDE Single Feature](Pics/distribution.png)
*Density overlap between real and synthetic distributions shows strong similarity.*

---

### 2. Multi-Feature Distribution Comparison

The following comparison highlights KDE distributions for two individual features, showing how well the synthetic data mimics real data across multiple dimensions.

![KDE Multi Features](.Pics/dis of features.png)
*Both features show well-aligned density peaks and shapes, demonstrating stable synthetic learning.*

---

### 3. Feature Correlation Preservation

These scatter plots show how feature correlations (e.g., Years vs. Salary) are maintained between real and synthetic datasets.  
While the synthetic data introduces minor variability, the underlying linear relationship remains intact.

![Feature Correlation](Pics/ex.png)
*Correlation structure is successfully preserved between real and generated sequences.*

---

### 4. Statistical Comparison Table

Quantitative metrics confirm the statistical similarity between real and synthetic datasets.  
The table below shows mean, standard deviation, and range differences, all within acceptable bounds (<10% for most metrics).

![Statistical Comparison](Pics/stats.png)

| Metric | Real Data | Synthetic Data | Difference | Relative Diff (%) |
|---------|------------|----------------|-------------|--------------------|
| Mean F1 | 0.454 | 0.418 | 0.036 | 7.99 |
| Std F1 | 0.244 | 0.234 | 0.010 | 4.14 |
| Mean F2 | 0.458 | 0.423 | 0.035 | 7.78 |
| Std F2 | 0.276 | 0.271 | 0.005 | 1.97 |

*Differences remain low, confirming strong alignment between synthetic and real data distributions.*

---

### 5. Temporal Dynamics Visualization (3D Trajectories)

To visually inspect the temporal behavior, 3D trajectory plots compare feature evolution over time.  
Both real and synthetic sequences follow similar paths, indicating that the model successfully captures temporal structure.

![3D Trajectories](Pics/3d.png)
*Temporal progression is well preserved; synthetic trajectories mimic real sequential dynamics.*

---

## Key Insights

- TimeGAN effectively merges **reconstruction** and **adversarial** objectives to produce realistic time-dependent data.  
- The **supervisor network** enhances temporal consistency and stabilizes training.  
- Statistical and visual evidence confirm that the synthetic data not only replicates distributions but also maintains **inter-feature dependencies** and **time-based relationships**.  
- Despite using a simple dataset, the results demonstrate scalability to more complex domains such as finance, healthcare, and IoT.

---

## Applications

TimeGAN is applicable to any domain requiring **synthetic time-series generation** with preserved structure and privacy:

- **Finance** – simulate credit scores, stock movements, or transaction patterns.  
- **Healthcare** – generate patient vital signals or medical sensor data.  
- **Industrial IoT** – produce realistic sensor logs for predictive maintenance.  
- **Behavioral modeling** – synthesize user activity, motion, or sensor sequences.

By creating statistically and temporally consistent synthetic data, TimeGAN allows organizations to **augment training datasets** or **share data responsibly** while preserving confidentiality.

---

## Visualization Summary

Below is a summary of visual results demonstrating model effectiveness:

1. **Distribution matching (KDE plots)** – overlapping densities.  
2. **Feature correlation preservation** – strong alignment between real and synthetic data.  
3. **Statistical parity** – low divergence in key metrics.  
4. **Temporal consistency (3D visualization)** – realistic sequential trajectories.

Together, these analyses confirm that the TimeGAN model **learns the generative process** rather than memorizing samples.

---

## Conclusion

This repository provides a full **PyTorch implementation** of the TimeGAN architecture — from model design and training to statistical and visual evaluation.  
It demonstrates that with the right combination of supervised, adversarial, and autoencoding objectives, it is possible to generate **high-quality, temporally coherent synthetic sequences** that preserve the dynamics and relationships of real-world data.

