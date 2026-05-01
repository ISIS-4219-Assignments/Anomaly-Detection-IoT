# Federated Learning — IoT Anomaly Detection: Experiment Analysis

## 1. Experimental Setup

| Parameter | Value |
|-----------|-------|
| Architectures | Vanilla (Dense), LSTM, Conv1D, Transformer |
| Devices | 8 (Distance, Flame\_Sensor, IR\_Receiver, phValue, Soil\_Moisture, Sound\_Sensor, Temperature\_and\_Humidity, Water\_Level) |
| Federated rounds | 7 |
| Local epochs | 20 max, early stopping patience = 3 (min\_delta = 1e-4) |
| Window size | 30 frames (LSTM, Conv1D, Transformer) / flat (Vanilla) |
| FL aggregation | Sample-weighted FedAvg — each device weighted by its training window count |
| Anomaly score | Median squared error over all non-batch dimensions |
| Threshold | 99th percentile of per-device validation reconstruction errors |
| Window labeling | A window is labeled **attack** if **any** of its 30 frames is an attack |

All models are trained as **unsupervised autoencoders on normal traffic only**. Attack data is never seen during training or threshold calibration. The detector exploits the principle that normal patterns are reconstructed well and anomalous ones are not.

---

## 2. Training Dynamics

Total training epochs (7 rounds × early-stopped local training):

| Device | Vanilla | LSTM | Conv1D | Transformer |
|--------|--------:|-----:|-------:|------------:|
| Distance | 54 | 69 | 124 | 76 |
| Flame\_Sensor | 54 | 82 | 87 | 84 |
| IR\_Receiver | 63 | 69 | 97 | 72 |
| phValue | 57 | 113 | 118 | 72 |
| Soil\_Moisture | 60 | 58 | 108 | 76 |
| Sound\_Sensor | 63 | 82 | 108 | 60 |
| Temperature\_and\_Humidity | 30 | 28 | 40 | 28 |
| Water\_Level | 65 | 69 | 93 | 66 |
| **Total** | **446** | **570** | **775** | **534** |
| **Avg per device** | **55.8** | **71.3** | **96.9** | **66.8** |

**Conv1D trains for the most epochs** across all devices, suggesting its loss landscape is harder to navigate but ultimately leads to better representations. The other three models are more frequently cut short by early stopping.

**Temperature\_and\_Humidity is the hardest device for all models.** All four architectures hit early stopping very aggressively here (28–40 total epochs vs 54–124 on other devices), indicating that the model cannot find a steadily improving minimum for this device's data distribution — a strong signal of difficult or noisy normal traffic.

---

## 3. Global Generalisation Test

Evaluated on a single file covering all 14 attack types pooled from all devices. This is the primary benchmark.

| Metric | Vanilla | LSTM | Conv1D | Transformer |
|--------|--------:|-----:|-------:|------------:|
| AUROC | 0.9962 | 0.9994 | **0.9999** | 0.9978 |
| PR-AUC | 0.9807 | 0.9985 | **0.9996** | 0.9882 |
| Precision | 0.9091 | 0.9743 | 0.9831 | **1.0000** |
| Recall | 0.9411 | 0.9958 | **0.9995** | 0.9448 |
| Balanced Accuracy | 0.9643 | 0.9961 | **0.9986** | 0.9724 |
| F1 (balanced) | 0.9634 | 0.9961 | **0.9986** | 0.9716 |
| TP | 27,037 | 29,011 | **29,120** | 27,525 |
| FN (missed attacks) | 1,691 | 123 | **14** | 1,609 |
| FP (false alarms) | 2,702 | 766 | 501 | **0** |
| TN | 213,798 | 215,299 | **215,564** | 216,065 |
| Total samples | 245,228 | 245,199 | 245,199 | 245,199 |
| Threshold | 0.0261 | 0.0001 | 0.0019 | 0.0095 |
| Normal Median MSE | 0.000517 | **0.000038** | 0.000678 | 0.005140 |
| Attack Median MSE | 453.2 | 0.00222 | 4.009 | 0.00989 |

**Conv1D is the best overall model.** It achieves the highest AUROC, PR-AUC, Recall, Balanced Accuracy, and F1, while missing only 14 attacks out of 29,134. The Transformer achieves a perfect precision of 1.0000 (literally zero false alarms on the global test) but at the cost of 1,609 missed attacks and three complete attack-type failures.

---

## 4. Network-Level Evaluation

Per-device thresholds applied; confusion matrices summed across all 8 devices.

| Metric | Vanilla | LSTM | Conv1D | Transformer |
|--------|--------:|-----:|-------:|------------:|
| AUROC (weighted) | 0.9976 | 0.9996 | **1.0000** | 0.9334 |
| PR-AUC (weighted) | 0.9832 | 0.9955 | **0.9992** | 0.8682 |
| Recall | 0.9821 | 0.9992 | **1.0000** | 0.9965 |
| FN | 2,373 | 108 | **3** | 463 |
| Precision | **0.9655** | 0.9033 | 0.9301 | 0.8000 |
| F1 | **0.9737** | 0.9488 | 0.9638 | 0.8875 |
| Balanced Accuracy | **0.9803** | 0.9667 | 0.9768 | 0.9215 |
| FP | **4,655** | 14,219 | 9,994 | 33,124 |
| TN | 211,870 | 201,668 | 205,893 | 182,763 |

At the network level, **Vanilla leads on F1, Precision, and Balanced Accuracy**. This is an artifact of one single device: Temperature\_and\_Humidity generates 12,168 LSTM false positives, 7,904 Conv1D false positives, and 33,124 Transformer false positives, while Vanilla only generates 3,014 there — because Vanilla trades recall (missing 1,645 attacks on that device) for fewer false alarms. The global test with a pooled threshold is the more honest benchmark.

**The Transformer collapses at the network level** (weighted AUROC 0.9334). Water\_Level contributes 0 TP and 0 FP (threshold too high — nothing flagged) while Temperature\_and\_Humidity contributes 0 TN and 33,124 FP (threshold too low — everything flagged). These two opposing failures expose a fundamental instability in the Transformer's threshold calibration.

---

## 5. Per-Attack-Type Breakdown (Global Test)

### Recall

| Attack Type | n | Vanilla | LSTM | Conv1D | Transformer |
|-------------|--:|--------:|-----:|-------:|------------:|
| Backdoor | 202 | **1.000** | 0.916 | **1.000** | 0.000 ❌ |
| DDoS\_HTTP | 604 | 0.484 ❌ | 0.987 | **0.987** | 0.949 |
| DDoS\_ICMP | 6,526 | **1.000** | 0.998 | **1.000** | 0.993 |
| DDoS\_TCP | 6,617 | **1.000** | 0.999 | **1.000** | 0.874 |
| DDoS\_UDP | 10,412 | **1.000** | 0.999 | **1.000** | 0.996 |
| MITM | 58 | **1.000** | 0.948 | **1.000** | 0.586 |
| OS\_Fingerprinting | 47 | **1.000** | 0.787 | **1.000** | 0.000 ❌ |
| Password | 3,147 | 0.618 ❌ | 0.998 | **1.000** | 0.986 |
| Port\_Scanning | 190 | **1.000** | 0.968 | **1.000** | 0.700 |
| Ransomware | 136 | 0.991 | 0.919 | **1.000** | 0.000 ❌ |
| SQL\_injection | 266 | 0.755 | 0.981 | **1.000** | 0.842 |
| Uploading | 185 | 0.545 ❌ | 0.935 | 0.984 | 0.773 |
| Vulnerability\_scanner | 591 | 0.984 | 0.988 | **1.000** | 0.953 |
| XSS | 153 | 0.476 ❌ | 0.974 | 0.980 | 0.797 |

### AUROC

| Attack Type | Vanilla | LSTM | Conv1D | Transformer |
|-------------|--------:|-----:|-------:|------------:|
| Backdoor | **1.000** | 0.982 | **1.000** | 0.897 |
| DDoS\_HTTP | 0.965 | 0.999 | **0.997** | 0.998 |
| DDoS\_ICMP | **1.000** | **1.000** | **1.000** | 0.999 |
| DDoS\_TCP | **1.000** | **1.000** | **1.000** | **1.000** |
| DDoS\_UDP | **1.000** | **1.000** | **1.000** | **1.000** |
| MITM | **1.000** | 0.994 | **1.000** | 0.977 |
| OS\_Fingerprinting | **1.000** | 0.935 | **1.000** | 0.939 |
| Password | 0.976 | **1.000** | **1.000** | 0.999 |
| Port\_Scanning | **1.000** | 0.999 | **1.000** | 0.978 |
| Ransomware | **1.000** | 0.990 | **1.000** | 0.904 |
| SQL\_injection | 0.983 | 0.998 | **1.000** | 0.994 |
| Uploading | 0.976 | 0.990 | **0.999** | 0.986 |
| Vulnerability\_scanner | 0.999 | 0.999 | **1.000** | 0.998 |
| XSS | 0.975 | 0.997 | **0.996** | 0.982 |

### Key observations

**Vanilla fails on 5 attack types** (marked ❌): DDoS\_HTTP, Password, SQL\_injection, Uploading, and XSS. These are application-layer attacks whose per-frame feature vectors are numerically close to normal traffic. Without temporal context, the vanilla autoencoder reconstructs them as accurately as normal traffic. DDoS variants succeed because they produce extreme numerical values (e.g., flood packet sizes, header anomalies) that are reconstructed poorly regardless of context.

**Transformer completely misses 3 attack types** — Backdoor, OS\_Fingerprinting, and Ransomware receive 0 TP each. Their AUROC scores (0.90, 0.94, 0.90) confirm the model *does* separate them in score space — the ranks are correct — but the threshold is placed above all of their anomaly scores. This is a threshold placement problem, not a representational failure. A lower percentile (e.g. 95th instead of 99th) would likely recover these at the cost of a small number of additional false alarms.

**Conv1D is the only model with no attack-type-level failures.** It achieves perfect or near-perfect recall across all 14 attack types including the hardest (OS\_Fingerprinting 100%, MITM 100%, Backdoor 100%, Ransomware 100%).

**LSTM's main weakness is OS\_Fingerprinting (78.7% recall).** This is a subtle network reconnaissance attack with only 47 test samples. 10 of 47 samples produce reconstruction errors below the threshold, suggesting the attack's traffic pattern is borderline to normal for the LSTM encoder.

---

## 6. MSE Separation Analysis

The ratio of attack median MSE to normal median MSE measures how clearly the autoencoder separates anomalies from normal traffic in reconstruction error space. A higher ratio means the threshold is easier to set and more robust to calibration errors.

| Model | Normal Median MSE | Attack Median MSE | Separation Ratio |
|-------|------------------:|------------------:|-----------------:|
| LSTM | 0.000038 | 0.00222 | 58× |
| Conv1D | 0.000678 | 4.009 | **5,912×** |
| Vanilla | 0.000517 | 453.2 | 876,000× ⚠️ |
| Transformer | 0.005140 | 0.00989 | **1.9×** |

**Conv1D achieves the most operationally useful separation** — 5,912× is large enough that the threshold is robust to minor distribution shifts between val and test data.

**Vanilla's 876,000× ratio is misleading.** It is driven by a handful of DDoS attacks with astronomically high MSE (DDoS\_TCP at the Soil\_Moisture device: ~38 billion). The median is pulled upward by these extremes while vanilla silently fails on attacks like DDoS\_HTTP (attack MSE ≈ 0.016, barely above normal MSE ≈ 0.000517). Vanilla's MSE distribution is highly bimodal: some attacks produce extreme errors, others produce errors indistinguishable from normal.

**The Transformer's 1.9× separation is critically low.** Normal and attack reconstruction errors are nearly identical, which is why threshold calibration is so fragile. This suggests the Transformer's bottleneck is under-constrained — it reconstructs both normal and attack traffic similarly well — or that the encoder has not learned to cleanly separate their latent representations.

---

## 7. Per-Device Deep Dive

### F1 (balanced) per device

| Device | Vanilla | LSTM | Conv1D | Transformer |
|--------|--------:|-----:|-------:|------------:|
| Distance | 0.9979 | 0.9958 | 0.9940 | 0.9948 |
| Flame\_Sensor | 0.9863 | 0.9874 | 0.9909 | **0.9993** |
| IR\_Receiver | 0.9941 | 0.9941 | 0.9950 | **0.9997** |
| phValue | 0.9832 | 0.9960 | 0.9959 | **0.9995** |
| Soil\_Moisture | 0.9936 | 0.9950 | 0.9936 | **0.9998** |
| Sound\_Sensor | 0.9944 | 0.9975 | 0.9966 | **0.9996** |
| Temperature\_and\_Humidity | 0.8865 | 0.8466 | 0.8938 | 0.6667 |
| Water\_Level | **1.0000** | 0.9910 | 0.9941 | 0.0000 ❌ |

On the six well-behaved devices the Transformer achieves the best F1 on five of them due to its perfect precision (0 false alarms). Temperature\_and\_Humidity and Water\_Level are where all models degrade for different reasons explained below.

### Full confusion matrices per device

(V = Vanilla, L = LSTM, C = Conv1D, T = Transformer)

#### Distance

| | Predicted Normal | Predicted Attack |
|-|:----------------:|:----------------:|
| **Actual Normal** | V: 23,146 / L: 22,985 / C: 22,874 / T: 23,151 | V: 92 / L: 166 / C: 277 / T: 0 |
| **Actual Attack** | V: 2 / L: 23 / C: 0 / T: 214 | V: 20,796 / L: 20,833 / C: 20,856 / T: 20,642 |

#### Flame\_Sensor

| | Predicted Normal | Predicted Attack |
|-|:----------------:|:----------------:|
| **Actual Normal** | V: 21,185 / L: 20,710 / C: 20,824 / T: 21,234 | V: 136 / L: 524 / C: 410 / T: 0 |
| **Actual Attack** | V: 313 / L: 20 / C: 0 / T: 21 | V: 14,579 / L: 14,930 / C: 14,950 / T: 14,929 |

#### IR\_Receiver

| | Predicted Normal | Predicted Attack |
|-|:----------------:|:----------------:|
| **Actual Normal** | V: 25,916 / L: 25,769 / C: 25,788 / T: 26,063 | V: 234 / L: 294 / C: 275 / T: 0 |
| **Actual Attack** | V: 29 / L: 17 / C: 0 / T: 9 | V: 16,958 / L: 17,028 / C: 17,045 / T: 17,036 |

#### phValue

| | Predicted Normal | Predicted Attack |
|-|:----------------:|:----------------:|
| **Actual Normal** | V: 14,036 / L: 14,014 / C: 13,995 / T: 14,113 | V: 135 / L: 99 / C: 118 / T: 0 |
| **Actual Attack** | V: 319 / L: 12 / C: 0 / T: 13 | V: 13,007 / L: 13,343 / C: 13,355 / T: 13,342 |

#### Soil\_Moisture

| | Predicted Normal | Predicted Attack |
|-|:----------------:|:----------------:|
| **Actual Normal** | V: 22,227 / L: 22,161 / C: 22,082 / T: 22,387 | V: 247 / L: 226 / C: 305 / T: 0 |
| **Actual Attack** | V: 58 / L: 11 / C: 3 / T: 11 | V: 23,660 / L: 23,765 / C: 23,773 / T: 23,765 |

#### Sound\_Sensor

| | Predicted Normal | Predicted Attack |
|-|:----------------:|:----------------:|
| **Actual Normal** | V: 30,007 / L: 30,126 / C: 30,051 / T: 30,257 | V: 337 / L: 131 / C: 206 / T: 0 |
| **Actual Attack** | V: 7 / L: 20 / C: 0 / T: 26 | V: 30,235 / L: 30,280 / C: 30,300 / T: 30,274 |

#### Temperature\_and\_Humidity — Problem device

| | Predicted Normal | Predicted Attack |
|-|:----------------:|:----------------:|
| **Actual Normal** | V: 30,197 / L: 20,956 / C: 25,220 / T: **0** | V: 3,014 / L: 12,168 / C: 7,904 / T: **33,124** |
| **Actual Attack** | V: **1,645** / L: 2 / C: 0 / T: 0 | V: 10,824 / L: 12,525 / C: 12,527 / T: 12,527 |

The sequence models (LSTM, Conv1D, Transformer) achieve near-perfect attack recall (99.98%–100%) but generate massive false alarm counts. The Transformer flags **every single normal sample** (TN = 0, FP = 33,124). The root cause is threshold collapse: the 99th-percentile of this device's validation reconstruction errors sits so close to the score distribution of both normal and attack test samples that the entire test set is classified as anomalous. Vanilla trades 1,645 missed attacks for a much lower false alarm count — landing in a different but also imperfect regime.

**Root cause:** Temperature\_and\_Humidity has high variance in its normal traffic, and the val split (15% of the device's capped rows) may not fully represent this variance. When the model encounters normal traffic at the edge of its learned distribution, it produces reconstruction errors above the aggressive threshold. A higher threshold percentile (99.5th–99.9th) for this specific device would dramatically reduce false alarms without sacrificing recall.

#### Water\_Level — Tiny attack set

| | Predicted Normal | Predicted Attack |
|-|:----------------:|:----------------:|
| **Actual Normal** | V: 45,156 / L: 44,947 / C: 45,059 / T: 45,558 | V: 460 / L: 611 / C: 499 / T: 0 |
| **Actual Attack** | V: 0 / L: 3 / C: 0 / T: **169** | V: 140 / L: 166 / C: 169 / T: 0 |

The test set contains only ~169 attack samples against ~45,557 normal samples — a 1:270 class imbalance. This makes precision inherently very low for all models regardless of architecture. The Transformer collapses completely here (AUROC = 0.5, 0 TP, 0 FP): its threshold is set at near-zero but the attack traffic MSE also rounds to zero, placing all samples below the threshold — nothing is ever flagged. The other three models catch all or nearly all attacks. **Results for Water\_Level should be treated as preliminary given the extremely small attack sample size.**

---

## 8. Threshold Analysis

| Device | Vanilla | LSTM | Conv1D | Transformer |
|--------|--------:|-----:|-------:|------------:|
| Distance | 0.0041 | 0.0001 | 0.0007 | ~0 |
| Flame\_Sensor | 0.0036 | 0.0001 | 0.0006 | ~0 |
| IR\_Receiver | 0.0049 | 0.0001 | 0.0006 | ~0 |
| phValue | 0.0036 | 0.0001 | 0.0008 | ~0 |
| Soil\_Moisture | 0.0065 | 0.0001 | 0.0006 | ~0 |
| Sound\_Sensor | 0.0045 | 0.0001 | 0.0006 | ~0 |
| Temperature\_and\_Humidity | 0.0046 | 0.0001 | 0.0007 | ~0 |
| Water\_Level | 0.0043 | 0.0001 | 0.0006 | ~0 |

**LSTM and Conv1D have the most consistent thresholds across all 8 devices** — LSTM at 0.0001 and Conv1D at 0.0006–0.0008 everywhere. This consistency means the threshold is portable: a value calibrated on one device generalises to others. It also reflects stable reconstruction quality that does not vary with device type.

**Vanilla thresholds vary by roughly 2×** (0.0036–0.0065), reflecting that different device normal traffic distributions produce different per-frame reconstruction error magnitudes.

**The Transformer threshold is effectively zero for all devices**, printed as 0.0000 due to display rounding. The actual values are on the order of 1e-6 to 1e-8, making the classifier extremely sensitive to tiny floating-point differences. Whether a device is classified as all-attack or all-normal depends on whether the test MSE lands just above or just below this near-zero boundary. This is not a viable operating regime.

---

## 9. Precision–Recall Tradeoff

For an IoT security detector, **recall is the primary objective** — a missed attack is a real security breach. False alarms are operationally costly but recoverable.

| Model | Global Recall | Global Precision | Operational assessment |
|-------|-------------:|-----------------:|------------------------|
| Conv1D | **0.9995** | 0.9831 | Best of both worlds |
| LSTM | 0.9958 | 0.9743 | Strong recall, minor precision cost |
| Transformer | 0.9448 | **1.0000** | Zero FP but misses 5.5% of attacks |
| Vanilla | 0.9411 | 0.9091 | Worst on both for application-layer attacks |

The Transformer's perfect precision is mathematically notable but operationally questionable. Missing 1,609 attacks while generating zero false alarms means it is tuned too conservatively. Shifting the threshold from the 99th to the 95th percentile would likely recover the three zero-recall attack types at the cost of a small number of additional false alarms — a trade well worth making in a security context.

---

## 10. The Transformer Threshold Problem — Root Cause

The Transformer's global val reconstruction MSE for normal traffic is **0.00514**, and attack MSE is **0.00989** — a gap of only 1.9×. By contrast, Conv1D achieves 5,912×. This means the Transformer's encoder does not produce substantially different latent representations for normal vs attack traffic.

Two contributing factors:

1. **Bottleneck compression.** The model uses a `latent_dim=16` dense bottleneck after `GlobalAveragePooling1D`. At only 16 dimensions, the code may not have enough capacity to accurately represent the diversity of normal patterns, causing normal MSE to rise toward attack MSE and compressing the gap.

2. **Decoder capacity.** The decoder is a Dense + Reshape + TimeDistributed path with no attention. The encoder learns positionally-aware representations (the fix added in this experiment), but the decoder may not have enough capacity to reconstruct fine-grained normal patterns from the 16-dimensional latent code, further inflating normal reconstruction error.

The consequence is that the 99th-percentile threshold falls inside the attack score distribution, and detection of any specific attack type depends on whether its individual MSE happens to land above this near-zero threshold. The three attack types with 0% recall are those whose attack MSE consistently falls below the threshold.

**Importantly, this is a calibration and capacity problem, not a fundamental architectural failure.** The positional encoding is in place, and AUROC scores for the failed attack types (Backdoor 0.897, OS\_Fingerprinting 0.939, Ransomware 0.904) confirm the model does rank them above many normal samples — it just cannot convert that ranking into a correct binary classification at the current threshold.

---

## 11. Overall Model Ranking

### 1. Conv1D — Best overall ✓

- **Global:** AUROC 0.9999, Recall 0.9995, F1 0.9986, FN = 14
- **Network:** AUROC 1.0000, Recall 1.0000, FN = 3
- Zero attack-type failures. Perfect recall on 11 of 14 attack types globally
- Widest MSE separation (5,912×) — threshold is robust and consistent (0.0006–0.0008 across all devices)
- Trains the most total epochs (avg 96.9), indicating thorough convergence
- **Recommended for deployment**

### 2. LSTM — Strong second ✓

- **Global:** AUROC 0.9994, Recall 0.9958, F1 0.9961, FN = 123
- Catches all but the hardest attack types; clear advantage over Vanilla on all subtle attacks
- Most consistent threshold of any model (exactly 0.0001 on every device)
- Weakness: OS\_Fingerprinting (78.7% recall) and high false alarm count on Temperature\_and\_Humidity
- **Safe fallback if Conv1D is unavailable**

### 3. Vanilla — Viable only for volume attacks ✗

- **Global:** AUROC 0.9962, Recall 0.9411, F1 0.9634, FN = 1,691
- Excellent on volume-based attacks with extreme feature values (DDoS variants with 100% recall)
- Catastrophically fails on 5 application-layer attack types (recall 48%–76%)
- Best network-level F1 (0.9737) — an artifact of fewer false alarms on Temperature\_and\_Humidity, not better detection
- **Not suitable as a standalone detector in a mixed-threat IoT environment**

### 4. Transformer — Not production-ready ✗

- **Global:** Precision 1.0000 (zero false alarms), but Recall 0.9448, FN = 1,609
- Three complete attack-type failures: Backdoor, OS\_Fingerprinting, Ransomware — all 0 TP
- Full network collapse: Water\_Level AUROC = 0.5 (random), Temperature\_and\_Humidity TN = 0
- Root cause: 1.9× MSE separation is too narrow for reliable threshold placement
- On the six well-behaved devices it achieves the best F1 of all models — the architecture is sound
- **Requires threshold recalibration or architectural changes to the bottleneck before deployment**

---

## 12. Findings and Recommendations

**Finding 1 — Temporal context is essential for application-layer attacks.**
Vanilla misses 24%–52% of DDoS\_HTTP, XSS, Password, SQL\_injection, and Uploading attacks. All three windowed models catch them at above 93% recall. The 30-frame window is the critical design choice enabling detection of attacks whose individual frames are indistinguishable from normal traffic.

**Finding 2 — Conv1D's global pooling bottleneck outperforms LSTM's recurrent bottleneck.**
For network traffic anomaly detection, global channel statistics across the window (what Conv1D extracts via GlobalAveragePooling) appear more discriminative than precise temporal ordering (what LSTM's hidden state captures). The 5,912× vs 58× MSE separation ratio is the quantitative result of this difference.

**Finding 3 — The Transformer needs architectural changes to the bottleneck.**
The positional encoding fix is working — the model is not order-blind — but the 1.9× MSE gap indicates insufficient compression of attack patterns. Recommended experiments: (a) reduce `latent_dim` from 16 to 8 to force more discriminative compression, (b) add a second transformer block to the encoder for richer representations, (c) try a CLS-token pooling strategy instead of GlobalAveragePooling, (d) lower the threshold percentile to 95th to recover the three zero-recall attack types immediately without retraining.

**Finding 4 — Temperature\_and\_Humidity requires a higher threshold percentile.**
The 99th-percentile threshold is too aggressive for this device across all sequence models. The val reconstruction error distribution is compressed, placing the threshold within the test normal distribution. Setting the threshold at the 99.5th or 99.9th percentile for this specific device would dramatically reduce false alarms (currently 3,014–33,124 FP) while preserving near-100% recall. A per-device adaptive percentile should be explored.

**Finding 5 — Water\_Level evaluation is inconclusive.**
With only ~169 attack samples against ~45,557 normal samples (1:270 imbalance), threshold-based metrics are dominated by noise. AUROC scores for Vanilla (0.9999), LSTM (0.9983), and Conv1D (1.0000) confirm those three models do learn a useful representation. More attack data is needed before conclusions can be drawn about this device.

**Finding 6 — Weighted FedAvg correctly reflects data heterogeneity.**
Devices have different training set sizes due to varying normal traffic CSV lengths under the 20% cap. Sample-weighted aggregation ensures larger devices (Sound\_Sensor, Soil\_Moisture) have proportionally more influence on the global model than smaller ones. This is the correct approach for a heterogeneous federation.

---

## 13. Conclusion

**Conv1D is the clear winner.** It is the only model achieving near-perfect recall across all 14 attack types, a stable and consistent threshold across all 8 devices, and the widest separation between normal and attack reconstruction errors. It is the recommended architecture for this IoT anomaly detection task.

LSTM is a reliable second choice. Vanilla is adequate only if the threat model excludes application-layer attacks. The Transformer, despite its theoretical advantages and strong performance on six devices, requires threshold recalibration or bottleneck redesign before it can be considered production-ready.

The two most impactful follow-up actions are: (1) address the Temperature\_and\_Humidity threshold problem with a device-specific higher percentile, and (2) investigate the Transformer bottleneck to improve MSE separation — either by reducing `latent_dim`, adding encoder depth, or adjusting the threshold percentile.
