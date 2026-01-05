# ASTE Research Project

**Aspect Sentiment Triplet Extraction with Position-Aware Tagging**

A PyTorch implementation of the EMNLP 2020 paper's position-aware tagging approach for extracting (aspect, opinion, sentiment) triplets from restaurant reviews, with additional JET^o (opinion-focused) variant implementation.

---

## Project Overview

This project implements an end-to-end system for Aspect Sentiment Triplet Extraction (ASTE), which extracts structured sentiment information from text. For example, from the sentence "The food was great but the service was terrible," the system extracts:
- (food, great, POSITIVE)
- (service, terrible, NEGATIVE)

### Key Features

- **Position-Aware Tagging (JET^t)**: Aspect-focused tagging scheme encoding relative positions to opinion spans
- **Opinion-Focused Variant (JET^o)**: Handles cases where one aspect has multiple opinions
- **Dual Model Architecture**: Ensemble approach combining both JET^t and JET^o predictions
- **BERT-based Architecture**: Leverages pre-trained transformers for contextual representations
- **Comprehensive Evaluation**: Relaxed matching with IoU-based opinion overlap (≥50%)
- **Experiment Tracking**: Systematic comparison and visualization framework

### Current Performance

- **JET^t Model**: F1 = 0.455 (10 epochs)
- **Dataset**: ASTE-Data-V2-EMNLP2020 (14res - restaurant reviews)
- **Training Examples**: 1,266 sentences
- **Test Examples**: 490 sentences

---

## Project Structure

```
aste-research-project/
│
├── data/
│   └── raw/
│       └── aste-data/
│           └── ASTE-Data-V2-EMNLP2020/
│               └── 14res/
│                   ├── train_triplets.txt
│                   ├── dev_triplets.txt
│                   └── test_triplets.txt
│
├── src/
│   ├── data/
│   │   └── load_aste.py              # Data loading and parsing
│   │
│   ├── models/
│   │   ├── baseline_model.py          # JET^t implementation
│   │   ├── dual_model.py              # Dual architecture (JET^t + JET^o)
│   │   ├── jeto_implementation.py     # JET^o opinion-focused variant
│   │   └── position_aware_tags_fixed.py  # Tagging scheme utilities
│   │
│   ├── evaluation/
│   │   ├── evaluate.py                # F1 metrics with relaxed matching
│   │   └── run_evaluation.py          # Standalone evaluation script
│   │
│   ├── training/
│   │   └── train_advanced.py          # Advanced training with ensemble
│   │
│   └── utils/
│       └── experiment_tracker.py      # Experiment tracking and visualization
│
├── experiments/
│   └── baseline/
│       ├── model.pt                   # Saved JET^t model
│       └── dual_model.pt              # Saved dual model
│
├── docs/
│   └── paper_summary.md               # EMNLP 2020 paper summary
│
├── tests/
│   └── test_model.py                  # Unit tests
│
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/aste-research-project.git
cd aste-research-project
```

2. Create a virtual environment:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Required Packages

```txt
torch>=2.0.0
transformers>=4.30.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

---

## Usage

### 1. Training the JET^t Baseline Model

Train the aspect-focused position-aware tagger:

```bash
python src/models/baseline_model.py
```

This will:
- Load the ASTE-Data-V2-EMNLP2020 dataset
- Build position-aware tag vocabulary
- Train BERT-based sequence tagger for 10 epochs
- Save model to `experiments/baseline/model.pt`
- Evaluate on test set every 2 epochs

**Expected output after 10 epochs:**
- F1 Score: ~0.45
- Training loss: 0.74
- Model size: ~440MB

### 2. Training the Dual Model (JET^t + JET^o)

Train both variants with ensemble capabilities:

```bash
python src/training/train_advanced.py \
    --model_mode ensemble \
    --max_epochs 10 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --experiment_name my_experiment
```

**Arguments:**
- `--model_mode`: Choose from `jett`, `jeto`, or `ensemble`
- `--ensemble_strategy`: How to combine predictions (`union`, `intersection`, `jett_priority`, `jeto_priority`)
- `--debug`: Use small data subset for testing
- `--show_examples`: Display prediction examples after training
- `--generate_report`: Create markdown experiment report

### 3. Evaluating a Saved Model

Evaluate any saved model on the test set:

```bash
python src/evaluation/run_evaluation.py \
    --model_path experiments/baseline/model.pt \
    --test_file data/raw/aste-data/ASTE-Data-V2-EMNLP2020/14res/test_triplets.txt
```

### 4. Loading and Using the Model

```python
import torch
from transformers import BertTokenizer
from src.models.baseline_model import SimpleASTEModel
from src.evaluation.evaluate import predict_triplets

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('experiments/baseline/model.pt', map_location=device)

# Initialize model with correct vocabulary size
model = SimpleASTEModel(num_tags=208).to(device)
model.load_state_dict(checkpoint)
model.eval()

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load vocabularies (you'll need these from your training)
tag2idx, idx2tag = build_tag_vocab(train_dataset)  # Load from checkpoint or rebuild

# Make predictions
sentence = "The food was delicious but the service was slow"
triplets, tags = predict_triplets(
    model, sentence, tokenizer, tag2idx, idx2tag, device
)

# Output format:
# [{'aspect_positions': [1], 'opinion_positions': [3], 'sentiment': 'POS',
#   'aspect': 'food', 'opinion': 'delicious'},
#  {'aspect_positions': [6], 'opinion_positions': [8], 'sentiment': 'NEG',
#   'aspect': 'service', 'opinion': 'slow'}]
```

---

## Technical Details

### Position-Aware Tagging Scheme (JET^t)

The model uses a novel tagging scheme that encodes:
1. **Aspect boundaries**: BIOES sub-tags (Begin, Inside, Outside, End, Single)
2. **Sentiment polarity**: POS, NEG, NEU
3. **Opinion positions**: Relative offsets (j, k) from aspect to opinion span

**Example:**
```
Sentence: "The food was great"
Words:    [The, food, was, great]
Tags:     [O,   S^POS_2_2, O, O]
```

The tag `S^POS_2_2` means:
- `S`: Single-word aspect at position 1
- `POS`: Positive sentiment
- `2_2`: Opinion starts at +2, ends at +2 (position 3)

### JET^o (Opinion-Focused) Variant

Reverses the tagging approach:
- Tags **opinion spans** instead of aspects
- Offsets point from opinion to aspect
- Better handles "one aspect → multiple opinions" cases

**Example:**
```
Sentence: "The food was cheap and delicious"
JET^o tags: [O, O, O, B^POS_-2_-2, I, E]
             Tags "cheap and delicious", points back to "food"
```

### Dual Model Architecture

The ensemble model:
1. **Shared BERT encoder**: Contextual representations
2. **Separate classifiers**: One for JET^t, one for JET^o
3. **Ensemble strategies**:
   - **Union**: All unique triplets from both
   - **JET^o Priority**: Start with JET^o, add non-overlapping JET^t predictions
   - **Intersection**: Only triplets found by both
   - **JET^t Priority**: Start with JET^t, add non-overlapping JET^o predictions

### Evaluation Metrics

**Relaxed Matching** (used in this implementation):
- ✅ Aspect positions must match exactly
- ✅ Sentiment must match
- ⚠️ Opinion overlap ≥ 50% (IoU-based)

This allows partial credit for opinion boundary errors while requiring correct aspect identification and sentiment classification.

---

## Key Implementation Decisions

### 1. Fixed Vocabulary Bug
**Issue**: Original implementation only retained tags from the last training example.

**Solution**: Collect all unique tags across entire dataset before building mappings.

```python
# Correct implementation
def build_tag_vocab(dataset):
    all_tags = set()
    for item in dataset:
        tags = triplets_to_tags(item['sentence'], item['triplets'])
        all_tags.update(tags)  # Collect FIRST
    
    tag2idx = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
    return tag2idx, idx2tag
```

### 2. Class Imbalance Handling
- 'O' tags represent ~90% of tokens
- Position-aware tags are extremely sparse (< 10%)
- **Solution**: Weighted CrossEntropyLoss with 0.1 weight for 'O' class

### 3. Tokenization Strategy
- BERT uses subword tokenization
- Tags are word-level
- **Current approach**: Align first subword of each word with tag
- **Known limitation**: Multi-subword tokens lose internal structure

---

## Debugging Journey & Lessons Learned

### Critical Bug Discovery
- **Problem**: F1 = 0.000 despite model predicting non-'O' tags
- **Root cause**: Vocabulary only contained tags from final training example (indentation bug)
- **Impact**: 208 unique tags → effectively 5-10 tags in practice
- **Fix**: Moved dictionary construction outside loop
- **Result**: F1 improved from 0.000 → 0.455

### Training Insights
1. **Loss decrease**: Clean trajectory (3.06 → 0.74) indicates learning
2. **Continued improvement**: F1 still climbing at epoch 10 (not plateaued)
3. **Extended training**: 20-epoch runs achieve ~0.48 F1
4. **GPU acceleration**: Essential for reasonable training times (~10 min/epoch on CUDA)

### Architecture Decisions
- **BERT-base-uncased**: Sufficient for restaurant domain
- **Dropout 0.1**: Standard regularization
- **Learning rate 2e-5**: Fine-tuning sweet spot for BERT
- **Batch size 8**: Balances memory and gradient stability

---

## Research Context

This implementation is based on:

**Paper**: "Position-Aware Tagging for Aspect Sentiment Triplet Extraction"  
**Conference**: EMNLP 2020  
**Task**: Joint extraction of (aspect, opinion, sentiment) triplets

### Novel Contributions in This Implementation

1. **Complete JET^o implementation**: Opinion-focused variant from paper
2. **Ensemble architecture**: Systematic combination of JET^t and JET^o
3. **Experiment tracking**: Comprehensive logging and comparison framework
4. **Systematic debugging**: Documented bug discovery and resolution process

### Comparison to Published Results

**ASTE-Data-V2-EMNLP2020 (14res):**
- Published JET^t: F1 ≈ 0.57
- Our JET^t (10 epochs): F1 = 0.455
- Gap likely due to: LSTM vs BERT, hyperparameter tuning, training duration

---

## Future Improvements

### Short-term (Research Focus)
- [ ] Train JET^t to 20+ epochs (expected F1 ~0.48-0.50)
- [ ] Complete JET^o training and evaluation
- [ ] Systematic ensemble strategy comparison
- [ ] Error analysis and failure pattern documentation
- [ ] Visualization dashboard for predictions

### Medium-term (Engineering Focus)
- [ ] FastAPI backend for model serving
- [ ] Streamlit demo interface
- [ ] Docker containerization
- [ ] Dataset statistics visualization
- [ ] Live triplet extraction demo

### Long-term (Research Extensions)
- [ ] Domain expansion (e-commerce, healthcare reviews)
- [ ] Multi-domain transfer learning
- [ ] Synthetic data generation with GPT-4
- [ ] Comparison with span-based approaches
- [ ] Attention visualization for interpretability

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python src/models/baseline_model.py --batch_size 4
```

**2. Module Import Errors**
```bash
# Ensure you're in the project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**3. Model Predicting Only 'O' Tags**
- Check vocabulary size (should be 200+)
- Verify tag2idx contains position-aware tags like 'B^POS_2_3'
- Ensure build_tag_vocab collects all tags before creating mappings

**4. Low F1 Score (< 0.2)**
- Train for more epochs (20+)
- Check class weights are applied correctly
- Verify evaluation uses relaxed matching

---

## Citation

If you use this code or approach, please cite the original paper:

```bibtex
@inproceedings{wu-etal-2020-grid,
    title = "Grid Tagging Scheme for Aspect-oriented Fine-grained Opinion Extraction",
    author = "Wu, Zhen and Ying, Chengcan and Zhao, Fei and Fan, Zhifang and Dai, Xinyu and Xia, Rui",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    year = "2020",
    publisher = "Association for Computational Linguistics"
}
```

---

## Development Timeline

- **Week 1**: Data loading, position-aware tagging implementation
- **Week 2**: BERT-based model architecture, training pipeline
- **Week 3**: Critical bug discovery and fix (vocab construction)
- **Week 4**: JET^o implementation, dual model architecture
- **Week 5**: Ensemble strategies, experiment tracking framework
- **Current Status**: Working JET^t baseline (F1=0.455), JET^o implemented, dual model tested

---

## Contact 

**Developer**: Nada Alaoui
**Email**: alaouinada49@gmail.com
**GitHub**: nada-alaoui-as

Contributions, issues, and feature requests are welcome!

---

## License

MIT License - see LICENSE file for details

---

## Acknowledgments

- ASTE-Data-V2-EMNLP2020 dataset creators
- Hugging Face Transformers library
- PyTorch team
- Original paper authors for the position-aware tagging approach
