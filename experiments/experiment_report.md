# ASTE Experiment Report

Generated: 2025-10-09 20:21:47

## Experiment Summary

| name                  | status    |   best_valid_f1 |   best_valid_precision |   best_valid_recall |
|:----------------------|:----------|----------------:|-----------------------:|--------------------:|
| JET_t_baseline        | completed |        0.737828 |               0.787979 |            0.667899 |
| JET_o_opinion_focused | completed |        0.714473 |               0.796372 |            0.686461 |

## Detailed Results

### JET_t_baseline (JET_t_baseline_20251009_202147)

**Configuration:**
- model: JET_t
- learning_rate: 2e-05
- batch_size: 8
- max_epochs: 10
- dropout: 0.1
- max_offset: 10

**Final Test Metrics:**
- test_f1: 0.7200
- test_precision: 0.7800
- test_recall: 0.6700

**Best Validation (Epoch 2):**
- f1: 0.7378
- precision: 0.7880
- recall: 0.6679

---

### JET_o_opinion_focused (JET_o_opinion_focused_20251009_202147)

**Configuration:**
- model: JET_o
- learning_rate: 2e-05
- batch_size: 8
- max_epochs: 10
- dropout: 0.1
- max_offset: 10

**Final Test Metrics:**
- test_f1: 0.7400
- test_precision: 0.7600
- test_recall: 0.7100

**Best Validation (Epoch 2):**
- f1: 0.7145
- precision: 0.7964
- recall: 0.6865

---

## Analysis

**Best Performing Model:** JET_t_baseline
- F1 Score: 0.7378

**Key Success Factors:**
- model: JET_t
- learning_rate: 2e-05
- batch_size: 8
- max_epochs: 10
- dropout: 0.1
- max_offset: 10