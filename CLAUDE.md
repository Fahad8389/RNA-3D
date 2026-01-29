# RNA 3D Structure Prediction

> *"If proteins are the workers of the cell, RNA is the messenger that got tired of just delivering mail and decided to do everything else too."*

## The Mission: Folding the Unfolded

Imagine you're given a string of letters—just A, U, G, and C—and someone asks you: *"What does this look like in 3D?"* That's essentially what we're doing here. We're predicting how RNA molecules twist, turn, and fold into intricate 3D shapes, armed with nothing but their sequence.

This project is our submission to the **Stanford RNA 3D Folding** Kaggle competition ($75,000 in prizes), where the goal is to solve one of biology's remaining grand challenges.

---

## Why Should You Care?

### The AlphaFold Gap

In 2020, DeepMind's AlphaFold basically "solved" protein structure prediction. The scientific world collectively lost its mind. Nobel Prizes were awarded. Champagne was popped.

But here's the thing: **RNA got left behind.**

While proteins have hundreds of thousands of known structures to learn from, RNA has maybe a few thousand. It's like trying to learn chess from 50 games versus 50,000 games. The scarcity of data makes RNA prediction significantly harder.

### Why RNA Matters

RNA isn't just the boring messenger molecule your high school biology class made it out to be:

- **mRNA vaccines** (hello, COVID shots) are literally injecting RNA into your body
- **CRISPR gene editing** uses guide RNAs to find and cut DNA
- **Riboswitches** in bacteria can turn genes on/off based on molecular signals
- **Long non-coding RNAs** regulate everything from cancer to brain development

Understanding how RNA folds = understanding how life's machinery works.

---

## The Competition At a Glance

| Aspect | Details |
|--------|---------|
| **Host** | Stanford University |
| **Prize Pool** | $75,000 |
| **Metric** | TM-score (0.0 to 1.0, higher = better) |
| **Input** | RNA sequence (AUGC letters) |
| **Output** | 3D coordinates (x, y, z) for each nucleotide |
| **Predictions** | 5 structures per sequence (best of 5 scored) |
| **Runtime** | ≤8 hours on Kaggle GPU |
| **Deadline** | September 24, 2025 |

### What's TM-score?

Think of it like this: if you predicted a structure and then tried to "overlay" it on top of the real structure, TM-score measures how well they match up. A score of 1.0 means perfect alignment. A score of 0.3 means you basically predicted a spaghetti monster when it should have been a pretzel.

The formula looks scary, but the intuition is simple: **closer atoms = higher score**, with some normalization so longer sequences don't have an unfair advantage.

---

## Project Structure

```
RNA-3D/
├── CLAUDE.md           # You are here! The project brain dump
├── README.md           # Quick start guide
├── configs/            # Hyperparameters, model configs
├── data/               # Competition data (gitignored)
├── models/             # Saved model weights (gitignored)
├── notebooks/          # Kaggle submission notebooks
└── src/                # Source code
    ├── data/           # Data loading and preprocessing
    ├── models/         # Model architectures
    ├── training/       # Training loops and utilities
    └── inference/      # Prediction and submission generation
```

---

## Technical Architecture

### The Pipeline (How Data Flows)

```
RNA Sequence → Tokenization → Feature Extraction → Neural Network → 3D Coordinates
     ↓              ↓               ↓                   ↓              ↓
  "AUGCGUA"    [0,1,2,3,2,1,0]   [embeddings,      [Transformer/    [x,y,z for
                                  MSA, etc.]        GNN layers]      each base]
```

### Key Components

#### 1. **Input Processing**
- Convert sequence letters (A, U, G, C) into numerical tokens
- Generate **Multiple Sequence Alignments (MSA)** if available—these show evolutionary patterns
- Compute structural features (base pairing probabilities, secondary structure predictions)

#### 2. **The Model** (RibonanzaNet-inspired)
The foundation is built on learnings from the previous Stanford Ribonanza competition:

- **Backbone**: Transformer architecture (attention is all you need, right?)
- **Structural Awareness**: Graph Neural Networks to capture spatial relationships
- **Multi-task Learning**: Predict multiple properties simultaneously

#### 3. **Output Heads**
- Predict (x, y, z) coordinates for the C1' atom of each nucleotide
- Generate 5 diverse structures using different sampling strategies

### Why These Technical Choices?

| Choice | Why |
|--------|-----|
| **Transformers** | RNA is a sequence problem. Attention mechanisms naturally capture long-range dependencies (a base at position 5 might pair with position 95) |
| **Graph Neural Networks** | Once we have distance predictions, GNNs help refine the 3D structure by "message passing" between nearby atoms |
| **Multi-task Learning** | Training on multiple related tasks (secondary structure, distances, angles) gives the model richer representations |
| **Ensemble of 5** | Structural prediction is uncertain. Multiple predictions hedge our bets. |

---

## Technologies Used

### Core Stack
- **Python 3.10+**: The lingua franca of ML
- **PyTorch**: Our deep learning framework (more flexible than TensorFlow for research)
- **PyTorch Lightning**: Keeps training code clean and organized
- **Weights & Biases**: Experiment tracking (never lose track of that one magical hyperparameter combo)

### Bioinformatics
- **Biopython**: Parsing sequences and structures
- **RNAfold** (ViennaRNA): Secondary structure prediction
- **US-align**: Structure alignment and TM-score calculation

### Infrastructure
- **Kaggle Notebooks**: Final submission environment
- **Docker**: Reproducible environments
- **GitHub Actions**: CI/CD for testing

---

## The Learning Journey

### Lessons from the Trenches

> *This section will grow as we build. Every bug squashed, every "aha!" moment, every 3 AM debugging session goes here.*

#### Lesson 1: Data is Everything (But There Isn't Much)
**The Problem**: Unlike ImageNet with millions of images, RNA structures number in the low thousands.

**What We Learned**:
- Data augmentation is critical (rotations, reflections, noise injection)
- Pre-training on related tasks (secondary structure, chemical probing data) gives a massive boost
- Synthetic data from physics-based simulations can supplement real data

#### Lesson 2: The Curse of Long Sequences
**The Problem**: Some RNAs are 1000+ nucleotides. Self-attention is O(n²).

**What We Learned**:
- Chunk-based processing with overlapping windows
- Linear attention variants (Performer, Linformer) as alternatives
- Hierarchical models: local attention + global pooling

#### Lesson 3: Coordinate Prediction is Tricky
**The Problem**: Predicting raw (x, y, z) coordinates is unstable—tiny errors compound.

**What We Learned**:
- Predict **distances** and **angles** instead, then reconstruct coordinates
- Use **invariant representations** (distances don't change if you rotate the molecule)
- **Iterative refinement**: start with a rough prediction, gradually improve

---

## Bugs We've Battled

> *A graveyard of bugs, so you don't repeat our mistakes.*

### Bug #1: [To be documented]
*Coming soon—we haven't made any mistakes yet. (Famous last words.)*

---

## Pitfalls to Avoid

### 1. **Overfitting to Public Leaderboard**
The public test set is small. A model that "wins" on 50 structures might completely fail on 500 new ones. Trust your local validation more than the leaderboard.

### 2. **Ignoring Physics**
ML models can predict physically impossible structures (atoms overlapping, bonds too long). Adding physics-based constraints or post-processing sanity checks helps.

### 3. **The "It Works on My Machine" Problem**
Kaggle's environment is different from your local setup. Test your submission notebook early and often.

### 4. **Memory Blowups**
RNA sequences can be long. Attention matrices grow quadratically. Monitor your GPU memory and use gradient checkpointing.

---

## How Good Engineers Think

### The Scientific Method Applies to Code

1. **Hypothesis**: "Adding MSA features will improve TM-score by 0.05"
2. **Experiment**: Implement, train, evaluate
3. **Analyze**: Did it work? Why or why not?
4. **Iterate**: Refine the hypothesis

Don't just throw techniques at the wall. Understand *why* something might work before trying it.

### Start Simple, Add Complexity

The temptation is to build the most sophisticated model possible. Resist it.

1. Start with a baseline (even a simple MLP)
2. Verify the pipeline works end-to-end
3. Add complexity incrementally
4. Measure each addition's impact

A working simple model beats a broken complex one every time.

### Version Control Everything

Not just code—data preprocessing steps, hyperparameters, random seeds. Future you will thank present you when you need to reproduce that one experiment from three weeks ago.

### Read the Winning Solutions

After every Kaggle competition, top teams share their approaches. This is gold. The Stanford Ribonanza competition (predecessor to this one) has detailed write-ups. Study them.

---

## Best Practices Checklist

- [ ] **Reproducibility**: Set random seeds everywhere
- [ ] **Validation**: Use time-based or structure-based splits, not random
- [ ] **Logging**: Track every experiment with W&B or similar
- [ ] **Testing**: Unit tests for data processing (it's where most bugs hide)
- [ ] **Documentation**: If you can't explain it, you don't understand it
- [ ] **Code Review**: Fresh eyes catch what tired eyes miss

---

## Resources & References

### Papers to Read
1. **RibonanzaNet**: The foundation model from the previous competition
2. **AlphaFold2**: Understand how the protein version works
3. **RNA-Puzzles assessments**: See what works and what doesn't

### Useful Links
- [Competition Page](https://www.kaggle.com/competitions/stanford-rna-3d-folding)
- [CASP16 RNA Results](https://predictioncenter.org/casp16/)
- [ViennaRNA Package](https://www.tbi.univie.ac.at/RNA/)

### Communities
- Kaggle Discussion Forums (competition-specific)
- r/MachineLearning and r/bioinformatics
- Twitter/X: Follow @RhijuDas (competition host)

---

## Progress Log

| Date | Milestone | Notes |
|------|-----------|-------|
| Jan 2025 | Project created | Initial setup, CLAUDE.md written |
| | | |

---

## Final Thoughts

This competition sits at the intersection of machine learning and fundamental biology. We're not just chasing prize money—we're contributing to a field that could unlock new medicines, reveal hidden biology, and push the boundaries of what AI can do with limited data.

The path will be messy. Models will fail. Bugs will appear at 2 AM. But that's the fun part.

Let's fold some RNA.

---

*"The important thing is not to stop questioning."* — Albert Einstein

*"Have you tried turning it off and on again?"* — Every engineer ever
