"""
Advanced Training Script for ASTE with JET^t, JET^o, and Ensemble
Complete pipeline with experiment tracking and visualization
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import argparse
import json
import sys
from pathlib import Path
import time
from typing import Dict, Tuple

# Add project paths
sys.path.append('src')

from data.load_aste import load_dataset
from models.position_aware_tags_fixed import triplets_to_tags, build_tag_vocab
from models.jeto_implementation import (
    triplets_to_tags_opinion_focused,
    build_tag_vocab_jeto,
    tags_to_triplets_opinion_focused
)
from models.dual_model import (
    ASTEDatasetDual,
    DualASTEModel,
    train_dual_model,
    predict_ensemble,
    collate_fn_dual
)
from evaluation.evaluate import calculate_f1, tags_to_triplets
from utils.experiment_tracker import ExperimentTracker, ModelComparator

def evaluate_dual_model(model, dataset, tokenizer, tag2idx_jett, idx2tag_jett,
                        tag2idx_jeto, idx2tag_jeto, device, 
                        ensemble_strategy='jeto_priority', max_samples=None):
    """
    Comprehensive evaluation of dual model.
    
    Returns:
        Dictionary with metrics for JET^t, JET^o, and ensemble
    """
    model.eval()
    
    # Initialize prediction collectors
    all_preds_jett = []
    all_preds_jeto = []
    all_preds_ensemble = []
    all_golds = []
    
    # Limit samples for faster evaluation if specified
    eval_dataset = dataset[:max_samples] if max_samples else dataset
    
    with torch.no_grad():
        for item in eval_dataset:
            sentence = item['sentence']
            gold_triplets = item['triplets']
            
            # Get predictions from all models
            pred_ensemble, pred_jett, pred_jeto = predict_ensemble(
                model, sentence, tokenizer,
                tag2idx_jett, idx2tag_jett,
                tag2idx_jeto, idx2tag_jeto,
                device, ensemble_strategy
            )
            
            all_preds_jett.extend(pred_jett)
            all_preds_jeto.extend(pred_jeto)
            all_preds_ensemble.extend(pred_ensemble)
            all_golds.extend(gold_triplets)
    
    # Calculate metrics for each approach
    metrics = {}
    
    # JET^t metrics
    if model.mode in ['jett', 'ensemble']:
        metrics_jett = calculate_f1(all_preds_jett, all_golds)
        metrics['jett'] = metrics_jett
    
    # JET^o metrics
    if model.mode in ['jeto', 'ensemble']:
        metrics_jeto = calculate_f1(all_preds_jeto, all_golds)
        metrics['jeto'] = metrics_jeto
    
    # Ensemble metrics
    if model.mode == 'ensemble':
        metrics_ensemble = calculate_f1(all_preds_ensemble, all_golds)
        metrics['ensemble'] = metrics_ensemble
    
    return metrics


def train_advanced_model(args):
    """
    Main training function with all advanced features.
    """
    print("="*60)
    print("Advanced ASTE Training with JET^o and Ensemble")
    print("="*60)
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(args.experiment_dir)
    
    # Load datasets
    print("\n1. Loading datasets...")
    train_data = load_dataset(args.train_file)
    val_data = load_dataset(args.val_file)
    test_data = load_dataset(args.test_file)
    
    # Subsample for faster experimentation if specified
    if args.debug:
        train_data = train_data[:100]
        val_data = val_data[:50]
        test_data = test_data[:50]
    
    print(f"   Train: {len(train_data)} samples")
    print(f"   Val: {len(val_data)} samples")
    print(f"   Test: {len(test_data)} samples")
    
    # Analyze dataset characteristics
    if args.analyze_data:
        print("\n2. Analyzing dataset characteristics...")
        ModelComparator.compare_tagging_schemes(train_data[:5])
    
    # Build vocabularies
    print("\n3. Building vocabularies...")
    tag2idx_jett, idx2tag_jett = build_tag_vocab(train_data)
    tag2idx_jeto, idx2tag_jeto = build_tag_vocab_jeto(train_data)
    
    print(f"   JET^t vocabulary: {len(tag2idx_jett)} tags")
    print(f"   JET^o vocabulary: {len(tag2idx_jeto)} tags")
    
    # Initialize tokenizer and datasets
    print("\n4. Preparing data loaders...")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    
    train_dataset = ASTEDatasetDual(train_data, tag2idx_jett, tag2idx_jeto, tokenizer)
    val_dataset = ASTEDatasetDual(val_data, tag2idx_jett, tag2idx_jeto, tokenizer)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_dual
    )
    
    # Initialize model
    print("\n5. Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"   Device: {device}")
    
    model = DualASTEModel(
        len(tag2idx_jett),
        len(tag2idx_jeto),
        mode=args.model_mode
    ).to(device)
    
    print(f"   Mode: {args.model_mode}")
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Start experiment tracking
    config = {
        'model_mode': args.model_mode,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'ensemble_strategy': args.ensemble_strategy,
        'bert_model': args.bert_model,
        'train_samples': len(train_data),
        'val_samples': len(val_data)
    }
    
    exp_name = f"{args.model_mode}_{args.experiment_name}"
    exp_id = tracker.start_experiment(exp_name, config)
    
    # Training loop
    print("\n6. Starting training...")
    print("-"*60)
    
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(args.max_epochs):
        print(f"\nEpoch {epoch + 1}/{args.max_epochs}")
        
        # Training phase
        model.train()
        start_time = time.time()
        
        train_loss = train_dual_model(
            model, train_loader, optimizer, device, tag2idx_jett, tag2idx_jeto, args.model_mode
        )
        
        epoch_time = time.time() - start_time
        print(f"Training loss: {train_loss:.4f} (Time: {epoch_time:.1f}s)")
        
        # Validation phase
        print("Evaluating on validation set...")
        val_metrics = evaluate_dual_model(
            model, val_data, tokenizer,
            tag2idx_jett, idx2tag_jett,
            tag2idx_jeto, idx2tag_jeto,
            device, args.ensemble_strategy,
            max_samples=100 if args.debug else None
        )
        
        # Log metrics
        tracker.log_metrics(exp_id, epoch, {'loss': train_loss}, phase='train')
        
        # Display validation results
        if args.model_mode == 'ensemble':
            print(f"Validation Results:")
            for mode_name in ['jett', 'jeto', 'ensemble']:
                if mode_name in val_metrics:
                    m = val_metrics[mode_name]
                    print(f"  {mode_name.upper():8} - F1: {m['f1']:.3f}, P: {m['precision']:.3f}, R: {m['recall']:.3f}")
                    tracker.log_metrics(exp_id, epoch, m, phase=f'valid_{mode_name}')
            
            # Use ensemble F1 for model selection
            current_f1 = val_metrics['ensemble']['f1']
        else:
            # Single model mode
            m = val_metrics.get(args.model_mode, {})
            print(f"Validation - F1: {m.get('f1', 0):.3f}, P: {m.get('precision', 0):.3f}, R: {m.get('recall', 0):.3f}")
            tracker.log_metrics(exp_id, epoch, m, phase='valid')
            current_f1 = m.get('f1', 0)
        
        # Check for improvement
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            print(f"New best F1: {best_val_f1:.3f} - Saving checkpoint...")
            tracker.save_checkpoint(
                exp_id, epoch,
                model.state_dict(),
                val_metrics,
                is_best=True
            )
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.patience}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    print("\n" + "="*60)
    print(f"Training completed! Best epoch: {best_epoch + 1} with F1: {best_val_f1:.3f}")
    print("="*60)
    
    # Load best model for final evaluation
    best_checkpoint_path = Path(tracker.experiments[exp_id]['dir']) / 'best_model.pt'
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded best model from epoch {checkpoint['epoch'] + 1}")
    
    # Final test evaluation
    print("\n7. Final evaluation on test set...")
    test_metrics = evaluate_dual_model(
        model, test_data, tokenizer,
        tag2idx_jett, idx2tag_jett,
        tag2idx_jeto, idx2tag_jeto,
        device, args.ensemble_strategy
    )
    
    print("\nTest Set Results:")
    print("-"*60)
    
    final_metrics = {}
    if args.model_mode == 'ensemble':
        for mode_name in ['jett', 'jeto', 'ensemble']:
            if mode_name in test_metrics:
                m = test_metrics[mode_name]
                print(f"{mode_name.upper():8} Performance:")
                print(f"  F1:        {m['f1']:.3f}")
                print(f"  Precision: {m['precision']:.3f}")
                print(f"  Recall:    {m['recall']:.3f}")
                print()
                
                # Store ensemble as final metrics
                if mode_name == 'ensemble':
                    final_metrics = {f"test_{k}": v for k, v in m.items()}
    else:
        m = test_metrics.get(args.model_mode, {})
        print(f"F1:        {m.get('f1', 0):.3f}")
        print(f"Precision: {m.get('precision', 0):.3f}")
        print(f"Recall:    {m.get('recall', 0):.3f}")
        final_metrics = {f"test_{k}": v for k, v in m.items()}
    
    # End experiment
    tracker.end_experiment(exp_id, final_metrics)
    
    # Save final model
    final_model_path = Path(args.experiment_dir) / exp_id / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'tag2idx_jett': tag2idx_jett,
        'tag2idx_jeto': tag2idx_jeto,
        'idx2tag_jett': idx2tag_jett,
        'idx2tag_jeto': idx2tag_jeto,
        'config': config,
        'test_metrics': test_metrics
    }, final_model_path)
    
    print(f"\nModel saved to: {final_model_path}")
    
    # Generate report
    if args.generate_report:
        report_path = Path(args.experiment_dir) / f"report_{exp_id}.md"
        report = tracker.generate_report([exp_id], save_path=str(report_path))
        print(f"Report saved to: {report_path}")
    
    # Demonstrate predictions on examples
    if args.show_examples:
        print("\n" + "="*60)
        print("Example Predictions")
        print("="*60)
        
        model.eval()
        for i, item in enumerate(test_data[:3]):
            sentence = item['sentence']
            gold = item['triplets']
            
            pred_ensemble, pred_jett, pred_jeto = predict_ensemble(
                model, sentence, tokenizer,
                tag2idx_jett, idx2tag_jett,
                tag2idx_jeto, idx2tag_jeto,
                device, args.ensemble_strategy
            )
            
            print(f"\nExample {i+1}:")
            print(f"Sentence: {sentence}")
            print(f"\nGold ({len(gold)} triplets):")
            for t in gold:
                print(f"  {t['aspect']} ← {t['opinion']} ({t['sentiment']})")
            
            if args.model_mode == 'ensemble':
                print(f"\nJET^t ({len(pred_jett)} triplets):")
                for t in pred_jett:
                    print(f"  {t['aspect']} ← {t['opinion']} ({t['sentiment']})")
                
                print(f"\nJET^o ({len(pred_jeto)} triplets):")
                for t in pred_jeto:
                    print(f"  {t['aspect']} ← {t['opinion']} ({t['sentiment']})")
                
                print(f"\nEnsemble ({len(pred_ensemble)} triplets):")
                for t in pred_ensemble:
                    print(f"  {t['aspect']} ← {t['opinion']} ({t['sentiment']})")
            else:
                predictions = pred_jett if args.model_mode == 'jett' else pred_jeto
                print(f"\nPredictions ({len(predictions)} triplets):")
                for t in predictions:
                    print(f"  {t['aspect']} ← {t['opinion']} ({t['sentiment']})")
    
    return exp_id, test_metrics


def main():
    parser = argparse.ArgumentParser(description='Advanced ASTE Training')
    
    # Data arguments
    parser.add_argument('--train_file', type=str, 
                       default='data/raw/aste-data/ASTE-Data-V2-EMNLP2020/14res/train_triplets.txt',
                       help='Path to training data')
    parser.add_argument('--val_file', type=str,
                       default='data/raw/aste-data/ASTE-Data-V2-EMNLP2020/14res/dev_triplets.txt',
                       help='Path to validation data')
    parser.add_argument('--test_file', type=str,
                       default='data/raw/aste-data/ASTE-Data-V2-EMNLP2020/14res/test_triplets.txt',
                       help='Path to test data')
    
    # Model arguments
    parser.add_argument('--model_mode', type=str, default='ensemble',
                       choices=['jett', 'jeto', 'ensemble'],
                       help='Model mode: jett, jeto, or ensemble')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased',
                       help='BERT model name')
    parser.add_argument('--ensemble_strategy', type=str, default='jeto_priority',
                       choices=['union', 'intersection', 'jett_priority', 'jeto_priority'],
                       help='Ensemble combination strategy')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=10,
                       help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=3,
                       help='Patience for early stopping')
    
    # Experiment arguments
    parser.add_argument('--experiment_name', type=str, default='experiment',
                       help='Name for this experiment')
    parser.add_argument('--experiment_dir', type=str, default='experiments',
                       help='Directory for experiment tracking')
    
    # Other arguments
    parser.add_argument('--debug', action='store_true',
                       help='Use small subset of data for debugging')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    parser.add_argument('--analyze_data', action='store_true',
                       help='Analyze dataset characteristics')
    parser.add_argument('--show_examples', action='store_true',
                       help='Show example predictions after training')
    parser.add_argument('--generate_report', action='store_true',
                       help='Generate experiment report')
    
    args = parser.parse_args()
    
    # Run training
    exp_id, test_metrics = train_advanced_model(args)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Experiment ID: {exp_id}")
    print("="*60)


if __name__ == "__main__":
    main()