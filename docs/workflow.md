# RNA 3D Structure Prediction: Project Workflow

> **Strategy**: Evolve RibonanzaNet2 with diffusion denoising (Baker Lab recipe for RNA)

---

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│    RNA Sequence ──────► RibonanzaNet2-Diffusion ──────► 3D Structure        │
│    "AUGCCU..."                                          (x,y,z coords)      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Project Workflow

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                           PHASE 1: DATA PREPARATION                           ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │         Download Datasets           │
                    ├─────────────────────────────────────┤
                    │  • Competition data (Kaggle)        │
                    │  • PDB structures (known 3D)        │
                    │  • RibonanzaNet2 pretrained weights │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │       Preprocess Structures         │
                    ├─────────────────────────────────────┤
                    │  • Parse PDB/mmCIF files            │
                    │  • Extract C1' atom coordinates     │
                    │  • Normalize coordinates            │
                    │  • Filter by quality/resolution     │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │         Create Data Splits          │
                    ├─────────────────────────────────────┤
                    │  • Train: structures before 2024    │
                    │  • Valid: structures 2024+          │
                    │  • Test: competition holdout        │
                    │  • Split by RNA family (no leak)    │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          PHASE 2: MODEL ARCHITECTURE                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│                         RibonanzaNet2-Diffusion                               │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                     ENCODER (from RibonanzaNet2)                        │  │
│  │                        [PRETRAINED - FROZEN/FINETUNE]                   │  │
│  │                                                                         │  │
│  │   Sequence ──► Embedding ──► Transformer Layers ──► RNA Features       │  │
│  │   "AUGCCU"      (256d)         (9 layers)           (per nucleotide)   │  │
│  │                                    │                                    │  │
│  │                                    ▼                                    │  │
│  │                           Pairwise Features                             │  │
│  │                        (nucleotide-nucleotide)                          │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                        │
│                                      ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                     DIFFUSION MODULE (NEW - TRAIN)                      │  │
│  │                                                                         │  │
│  │   Noisy Coords ──►┌──────────────────────┐                              │  │
│  │   (x,y,z + noise) │                      │                              │  │
│  │                   │   Denoising Network  │──► Predicted Clean Coords    │  │
│  │   RNA Features ──►│   (learns to remove  │    (x,y,z)                   │  │
│  │   (from encoder)  │    noise)            │                              │  │
│  │                   │                      │                              │  │
│  │   Timestep t ────►└──────────────────────┘                              │  │
│  │   (noise level)                                                         │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                            PHASE 3: TRAINING                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                           DIFFUSION TRAINING LOOP                             │
│                                                                               │
│   For each RNA with known structure:                                          │
│                                                                               │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                  │
│   │   Clean     │      │    Add      │      │   Noisy     │                  │
│   │ Structure   │─────►│   Noise     │─────►│ Structure   │                  │
│   │  (x,y,z)    │      │  (level t)  │      │ (x,y,z)+ε   │                  │
│   └─────────────┘      └─────────────┘      └─────────────┘                  │
│                                                    │                          │
│                                                    ▼                          │
│                                             ┌─────────────┐                   │
│                                             │   Model     │                   │
│                        Sequence ───────────►│  Predicts   │                   │
│                                             │   Clean     │                   │
│                                             └─────────────┘                   │
│                                                    │                          │
│                                                    ▼                          │
│   ┌─────────────┐                          ┌─────────────┐                   │
│   │   Clean     │                          │  Predicted  │                   │
│   │ Structure   │◄─── Compare (Loss) ─────►│   Clean     │                   │
│   │  (target)   │                          │ Structure   │                   │
│   └─────────────┘                          └─────────────┘                   │
│                                                    │                          │
│                              Backpropagate & Update Weights                   │
│                                                                               │
│   Repeat for t = 1, 2, 3, ... 200 (all noise levels)                         │
│   Repeat for all training RNAs                                                │
│   Repeat for many epochs                                                      │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                            PHASE 4: INFERENCE                                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                        GENERATING 3D STRUCTURE                                │
│                                                                               │
│   Input: New RNA sequence (no known structure)                                │
│                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │  Step 0: Initialize with pure random noise                          │    │
│   │                                                                     │    │
│   │     Sequence: "AUGCCU..."                                           │    │
│   │     Coords:   random (x,y,z) for each nucleotide                   │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                        │
│                                      ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │  Steps 1-200: Iterative Denoising                                   │    │
│   │                                                                     │    │
│   │     t=200      t=199      t=198              t=1       t=0          │    │
│   │    ┌─────┐    ┌─────┐    ┌─────┐           ┌─────┐   ┌─────┐       │    │
│   │    │noise│───►│ less│───►│ less│───► ... ─►│almost│──►│clean│       │    │
│   │    │     │    │noise│    │noise│           │clean │   │     │       │    │
│   │    └─────┘    └─────┘    └─────┘           └─────┘   └─────┘       │    │
│   │                                                                     │    │
│   │     Each step: Model predicts slightly cleaner structure            │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                        │
│                                      ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │  Output: Predicted 3D coordinates                                   │    │
│   │                                                                     │    │
│   │     A ──► (1.2, 3.4, 5.6)                                          │    │
│   │     U ──► (1.5, 3.1, 5.2)                                          │    │
│   │     G ──► (2.0, 2.8, 4.9)                                          │    │
│   │     ...                                                             │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│   Repeat 5 times with different random seeds ──► 5 diverse predictions       │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                           PHASE 5: SUBMISSION                                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │      Format Predictions             │
                    ├─────────────────────────────────────┤
                    │  • 5 structures per RNA sequence    │
                    │  • CSV with (x, y, z) per nucleotide│
                    │  • Match submission format          │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │      Submit to Kaggle               │
                    ├─────────────────────────────────────┤
                    │  • Upload submission.csv            │
                    │  • Get TM-score on public set       │
                    │  • Final eval on private set        │
                    └─────────────────────────────────────┘
```

---

## Key Components Detail

### A. Noise Schedule

```
t=0   ────────────────────────────────────────────── t=200
│                                                      │
▼                                                      ▼
Clean                                              Pure Noise
Structure                                          (Gaussian)

Noise increases linearly or with cosine schedule
```

### B. Denoising Network Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DENOISING NETWORK                        │
│                                                             │
│   Inputs:                                                   │
│   ├── Noisy coordinates: (N x 3) where N = sequence length │
│   ├── Sequence features: (N x 256) from RibonanzaNet2      │
│   ├── Pairwise features: (N x N x 64) from RibonanzaNet2   │
│   └── Timestep embedding: (1 x 256) encodes noise level    │
│                                                             │
│   Architecture options:                                     │
│   ├── Option 1: Graph Neural Network (like RNAgrail)       │
│   ├── Option 2: SE(3) Equivariant Network (like FrameFlow) │
│   └── Option 3: Transformer + IPA (like AlphaFold)         │
│                                                             │
│   Output:                                                   │
│   └── Predicted clean coordinates: (N x 3)                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### C. Loss Function

```
Loss = MSE(predicted_coords, actual_clean_coords)

Or more sophisticated:
Loss = MSE(coords) + λ₁·DistanceLoss + λ₂·AngleLoss + λ₃·ClashLoss
       └─ position ─┘  └─ pairwise ──┘  └─ local ──┘  └─ physics ─┘
```

---

## Project File Structure

```
RNA-3D/
├── CLAUDE.md                    # Project instructions
├── docs/
│   └── workflow.md              # This file
├── configs/
│   ├── model.yaml               # Model hyperparameters
│   ├── training.yaml            # Training settings
│   └── diffusion.yaml           # Noise schedule, timesteps
├── data/
│   ├── raw/                     # Downloaded data
│   ├── processed/               # Cleaned structures
│   └── splits/                  # Train/valid/test
├── src/
│   ├── data/
│   │   ├── dataset.py           # PyTorch dataset
│   │   ├── preprocessing.py     # PDB parsing
│   │   └── augmentation.py      # Data augmentation
│   ├── models/
│   │   ├── ribonanzanet.py      # Load pretrained encoder
│   │   ├── diffusion.py         # Diffusion module
│   │   ├── denoiser.py          # Denoising network
│   │   └── full_model.py        # Combined model
│   ├── training/
│   │   ├── trainer.py           # Training loop
│   │   ├── losses.py            # Loss functions
│   │   └── scheduler.py         # Noise schedule
│   └── inference/
│       ├── generate.py          # Structure generation
│       ├── sample.py            # Sampling strategies
│       └── submission.py        # Format for Kaggle
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_submission.ipynb      # Kaggle notebook
└── scripts/
    ├── download_data.sh
    ├── train.py
    └── predict.py
```

---

## Timeline

```
Week 1-2:   Data preparation + understand RibonanzaNet2 code
Week 3-4:   Implement diffusion module
Week 5-6:   Training + experiments
Week 7-8:   Optimization + submission tuning
Week 9+:    Iterate based on leaderboard feedback
```

---

## Success Metrics

| Metric | Target | Measured On |
|--------|--------|-------------|
| TM-score (local validation) | > 0.5 | Held-out 2024+ structures |
| TM-score (public LB) | > 0.6 | Kaggle public test |
| TM-score (private LB) | Top 10% | Final competition ranking |
| Inference time | < 8 hours | Full test set on Kaggle GPU |

---

---

## Competition Requirements Compliance

### Submission Format

```csv
ID,resname,resid,x_1,y_1,z_1,x_2,y_2,z_2,x_3,y_3,z_3,x_4,y_4,z_4,x_5,y_5,z_5
R1107_1,G,1,-7.561,9.392,9.361,-7.2,9.1,9.0,...
R1107_1,A,2,-8.02,11.014,14.606,-7.8,10.5,14.2,...
```

- **ID**: sequence_id (e.g., R1107_1)
- **resname**: nucleotide type (A, U, G, C)
- **resid**: residue number
- **x_1,y_1,z_1 ... x_5,y_5,z_5**: 5 predictions of C1' atom coordinates

### Important: C1' Atom Only

```
Each nucleotide has ~20-30 atoms
Competition only wants C1' atom (1 per nucleotide)
C1' = carbon connecting sugar to base (backbone position)
```

### Rules Compliance Checklist

| Rule | Requirement | Our Plan | Status |
|------|-------------|----------|--------|
| Runtime | ≤ 8 hours GPU/CPU | Designed for this | ✅ |
| Internet | Disabled during submission | Attach weights as Kaggle dataset | ✅ |
| External data | Must be publicly available | Using public PDB + RibonanzaNet2 | ✅ |
| Pretrained models | Allowed if public | RibonanzaNet2 is public | ✅ |
| Submissions | 5 per day, 2 final | No issue | ✅ |
| Winner license | Open source (OSI-approved) | Will use MIT/Apache | ✅ |
| No privileged access | Can't use private lab data | Using only public data | ✅ |

### Winner Obligations (If We Win)

Must provide:
- Training code
- Inference code
- Computational environment description
- Detailed methodology (architecture, preprocessing, loss, hyperparameters)

---

## Competitive Differentiation

### The Challenge

Everyone has access to:
- Same public PDB data
- Same RibonanzaNet2 pretrained model
- Same diffusion papers/methods
- Same competition data

### How to Stand Out (Next Session Focus)

| Factor | Potential Edge |
|--------|----------------|
| **Training data curation** | Better quality filtering, augmentation |
| **Noise schedule** | Custom schedule for RNA (not protein) |
| **Loss function** | Physics-informed losses (distances, angles, clashes) |
| **Architecture tweaks** | RNA-specific modifications to denoiser |
| **Self-conditioning** | Use previous predictions (like RFdiffusion) |
| **Ensemble** | Multiple models, different seeds |
| **Post-processing** | Physics refinement after generation |
| **Hybrid approach** | Combine diffusion + template search |

### Key Insight

```
Idea alone:        10% of winning
Execution:         50% of winning
Iteration speed:   30% of winning
Luck:              10% of winning

Many will try diffusion. Few will execute it well.
```

---

## Next Steps

1. [ ] Download competition data and RibonanzaNet2 weights
2. [ ] Set up project structure
3. [ ] Implement data preprocessing pipeline
4. [ ] Load and test RibonanzaNet2 encoder
5. [ ] Implement diffusion training loop
6. [ ] **Explore differentiation strategies** (next session)
7. [ ] Train and evaluate
8. [ ] Submit to Kaggle

---

## Session Log

### Session 1 (Current)
- ✅ Researched Part 1 winners (template-based won, RibonanzaNet foundation)
- ✅ Studied RibonanzaNet architecture (11M → 100M params for v2)
- ✅ Researched diffusion models for RNA (RNAgrail, RNA-FrameFlow)
- ✅ Decided strategy: RibonanzaNet2 + Diffusion (Baker Lab recipe for RNA)
- ✅ Created workflow diagram
- ✅ Verified competition rules compliance
- ✅ Identified need for differentiation strategies

### Session 2 (Next)
- [ ] Deep dive into differentiation strategies
- [ ] Set up development environment
- [ ] Download data and pretrained weights
- [ ] Start implementation
