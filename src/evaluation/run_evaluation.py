"""
Standalone evaluation script for trained ASTE models
Useful for evaluating saved models without retraining
"""

import torch
import argparse
import json
from pathlib import Path
import sys

sys.path.append('src/data')
sys.path.append('src/models')
sys.path.append('src/evaluation')

from load_aste import load_dataset
from position_aware_tags_fixed import build_tag_vocab
from baseline_model import SimpleASTEModel
from evaluate import evaluate_model
from transformers import BertTokenizer


def evaluate_saved_model(model_path, test_file, train_file=None, max_samples=None):
    """
    Evaluate a saved model on test data.
    
    Args:
        model_path: Path to saved model file
        test_file: Path to test dataset
        train_file: Path to training dataset (for vocabulary)
        max_samples: Limit evaluation to first N samples (None = all)
    """
    print("="*60)
    print("ASTE Model Evaluation")
    print("="*60)
    
    # Check if model exists
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load test data
    print(f"\n1. Loading test data from: {test_file}")
    test_dataset = load_dataset(test_file)
    
    if max_samples:
        test_dataset = test_dataset[:max_samples]
        print(f"   Using first {max_samples} examples")
    else:
        print(f"   Loaded {len(test_dataset)} test examples")
    
    # Load vocabulary
    if train_file:
        print(f"\n2. Building vocabulary from: {train_file}")
        train_dataset = load_dataset(train_file)
    else:
        # If no train file provided, try to load from checkpoint
        print("\n2. Loading vocabulary from checkpoint...")
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'tag2idx' in checkpoint:
            tag2idx = checkpoint['tag2idx']
            idx2tag = checkpoint['idx2tag']
        else:
            raise ValueError("No vocabulary found in checkpoint and no train file provided")
    
    if train_file:
        from position_aware_tags_fixed import build_tag_vocab
        tag2idx, idx2tag = build_tag_vocab(train_dataset)
    
    num_tags = len(tag2idx)
    print(f"   Vocabulary size: {num_tags} tags")
    
    # Load model
    print(f"\n3. Loading model from: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if it's a full checkpoint or just state dict
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        # Full checkpoint with metadata
        model = SimpleASTEModel(num_tags).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   Model loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        # Just state dict
        model = SimpleASTEModel(num_tags).to(device)
        model.load_state_dict(checkpoint)
        print("   Model loaded from state dict")
    
    print(f"   Device: {device}")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Evaluate
    print("\n4. Running evaluation...")
    metrics = evaluate_model(model, test_dataset, tokenizer, tag2idx, idx2tag, device)
    
    # Display results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Samples evaluated: {len(test_dataset)}")
    print(f"\nPerformance Metrics:")
    print(f"  F1 Score:    {metrics['f1']:.3f}")
    print(f"  Precision:   {metrics['precision']:.3f}")
    print(f"  Recall:      {metrics['recall']:.3f}")
    print(f"\nDetailed Counts:")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained ASTE model')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved model file')
    parser.add_argument('--test_file', type=str,
                       default='data/raw/aste-data/ASTE-Data-V2-EMNLP2020/14res/test_triplets.txt',
                       help='Path to test dataset')
    parser.add_argument('--train_file', type=str,
                       default='data/raw/aste-data/ASTE-Data-V2-EMNLP2020/14res/train_triplets.txt',
                       help='Path to training dataset (for vocabulary)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Limit evaluation to N samples')
    parser.add_argument('--save_results', type=str, default=None,
                       help='Path to save results JSON')
    
    args = parser.parse_args()
    
    # Run evaluation
    metrics = evaluate_saved_model(
        args.model_path,
        args.test_file,
        args.train_file,
        args.max_samples
    )
    
    # Save results if requested
    if args.save_results:
        results = {
            'model_path': args.model_path,
            'test_file': args.test_file,
            'num_samples': args.max_samples or 'all',
            'metrics': metrics
        }
        
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.save_results}")


if __name__ == "__main__":
    # If no arguments provided, run with defaults (backward compatibility)
    import sys
    if len(sys.argv) == 1:
        print("Running with default arguments...")
        print("For custom evaluation, use: python run_evaluation.py --help\n")
        
        # Default evaluation
        evaluate_saved_model(
            model_path='experiments/baseline/model.pt',
            test_file='data/raw/aste-data/ASTE-Data-V2-EMNLP2020/14res/test_triplets.txt',
            train_file='data/raw/aste-data/ASTE-Data-V2-EMNLP2020/14res/train_triplets.txt',
            max_samples=None  # Evaluate all samples by default
        )
    else:
        main()