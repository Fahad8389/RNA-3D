# RNA 3D Structure Prediction

Predicting 3D structures of RNA molecules from sequence alone.

**Competition**: [Stanford RNA 3D Folding](https://www.kaggle.com/competitions/stanford-rna-3d-folding) (Kaggle, $75K prizes)

## Quick Start

```bash
# Clone the repo
git clone https://github.com/Fahad8389/RNA-3D.git
cd RNA-3D

# Install dependencies
pip install -r requirements.txt

# Download competition data (requires Kaggle API)
kaggle competitions download -c stanford-rna-3d-folding -p data/
```

## Project Overview

See [CLAUDE.md](CLAUDE.md) for the full project documentation, including:
- Competition details and scoring
- Technical architecture
- Lessons learned
- Best practices

## Structure

```
├── CLAUDE.md       # Detailed project documentation
├── configs/        # Model and training configurations
├── data/           # Competition data (gitignored)
├── models/         # Saved weights (gitignored)
├── notebooks/      # Kaggle submission notebooks
└── src/            # Source code
```

## Competition Task

**Input**: RNA sequence (e.g., `AUGCGUACGUA...`)
**Output**: 3D coordinates (x, y, z) for each nucleotide's C1' atom
**Metric**: TM-score (0-1, higher is better)

## License

MIT
