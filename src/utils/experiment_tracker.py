"""
Comprehensive Experiment Framework for ASTE Research
Tracks experiments, compares models, generates reports
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path


class ExperimentTracker:
    """
    Track and compare experiments for ASTE research.
    Features:
    - Automatic experiment logging
    - Hyperparameter tracking
    - Performance metrics comparison
    - Visualization generation
    """
    
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.experiments_file = self.base_dir / "experiments.json"
        self.experiments = self.load_experiments()
    
    def load_experiments(self) -> Dict:
        """Load existing experiments or create new tracking file"""
        if self.experiments_file.exists():
            with open(self.experiments_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_experiments(self):
        """Save experiments to disk"""
        with open(self.experiments_file, 'w') as f:
            json.dump(self.experiments, f, indent=2, default=str)
    
    def start_experiment(self, name: str, config: Dict) -> str:
        """
        Start a new experiment.
        
        Args:
            name: Experiment name
            config: Configuration dictionary including model params, hyperparams, etc.
        
        Returns:
            Experiment ID
        """
        exp_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_dir = self.base_dir / exp_id
        exp_dir.mkdir(exist_ok=True)
        
        self.experiments[exp_id] = {
            'name': name,
            'config': config,
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'metrics': {},
            'checkpoints': [],
            'dir': str(exp_dir)
        }
        
        self.save_experiments()
        print(f"Started experiment: {exp_id}")
        return exp_id
    
    def log_metrics(self, exp_id: str, epoch: int, metrics: Dict, phase: str = 'train'):
        """
        Log metrics for an experiment.
        
        Args:
            exp_id: Experiment ID
            epoch: Current epoch
            metrics: Dictionary of metrics
            phase: 'train', 'valid', or 'test'
        """
        if exp_id not in self.experiments:
            raise ValueError(f"Experiment {exp_id} not found")
        
        if phase not in self.experiments[exp_id]['metrics']:
            self.experiments[exp_id]['metrics'][phase] = []
        
        metrics_entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        self.experiments[exp_id]['metrics'][phase].append(metrics_entry)
        self.save_experiments()
    
    def save_checkpoint(self, exp_id: str, epoch: int, model_state: Dict, 
                       metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        if exp_id not in self.experiments:
            raise ValueError(f"Experiment {exp_id} not found")
        
        exp_dir = Path(self.experiments[exp_id]['dir'])
        
        # Save checkpoint
        checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
        if is_best:
            checkpoint_name = "best_model.pt"
        
        checkpoint_path = exp_dir / checkpoint_name
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        
        self.experiments[exp_id]['checkpoints'].append({
            'epoch': epoch,
            'path': str(checkpoint_path),
            'metrics': metrics,
            'is_best': is_best
        })
        
        self.save_experiments()
    
    def end_experiment(self, exp_id: str, final_metrics: Optional[Dict] = None):
        """Mark experiment as completed"""
        if exp_id not in self.experiments:
            raise ValueError(f"Experiment {exp_id} not found")
        
        self.experiments[exp_id]['status'] = 'completed'
        self.experiments[exp_id]['end_time'] = datetime.now().isoformat()
        
        if final_metrics:
            self.experiments[exp_id]['final_metrics'] = final_metrics
        
        self.save_experiments()
        print(f"Completed experiment: {exp_id}")
    
    def compare_experiments(self, exp_ids: List[str], metric: str = 'f1') -> pd.DataFrame:
        """
        Compare multiple experiments.
        
        Args:
            exp_ids: List of experiment IDs to compare
            metric: Metric to compare
        
        Returns:
            DataFrame with comparison results
        """
        comparison = []
        
        for exp_id in exp_ids:
            if exp_id not in self.experiments:
                print(f"Warning: Experiment {exp_id} not found")
                continue
            
            exp = self.experiments[exp_id]
            row = {
                'experiment': exp_id,
                'name': exp['name'],
                'status': exp['status']
            }
            
            # Add config details
            if 'config' in exp:
                row.update({
                    f"config_{k}": v for k, v in exp['config'].items()
                    if isinstance(v, (int, float, str, bool))
                })
            
            # Add best metrics
            if 'final_metrics' in exp:
                row.update({f"final_{k}": v for k, v in exp['final_metrics'].items()})
            
            # Find best validation metrics
            if 'valid' in exp.get('metrics', {}):
                valid_metrics = exp['metrics']['valid']
                if valid_metrics:
                    best_valid = max(valid_metrics, key=lambda x: x.get(metric, 0))
                    row.update({f"best_valid_{k}": v for k, v in best_valid.items() 
                               if k not in ['epoch', 'timestamp']})
            
            comparison.append(row)
        
        return pd.DataFrame(comparison)
    
    def plot_learning_curves(self, exp_ids: List[str], metrics: List[str] = ['f1', 'precision', 'recall']):
        """Plot learning curves for multiple experiments"""
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        colors = sns.color_palette("husl", len(exp_ids))
        
        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx]
            
            for exp_idx, exp_id in enumerate(exp_ids):
                if exp_id not in self.experiments:
                    continue
                
                exp = self.experiments[exp_id]
                
                # Plot training curve
                if 'train' in exp.get('metrics', {}):
                    train_data = exp['metrics']['train']
                    epochs = [d['epoch'] for d in train_data if metric in d]
                    values = [d[metric] for d in train_data if metric in d]
                    if epochs:
                        ax.plot(epochs, values, label=f"{exp['name']} (train)", 
                               color=colors[exp_idx], linestyle='--', alpha=0.7)
                
                # Plot validation curve
                if 'valid' in exp.get('metrics', {}):
                    valid_data = exp['metrics']['valid']
                    epochs = [d['epoch'] for d in valid_data if metric in d]
                    values = [d[metric] for d in valid_data if metric in d]
                    if epochs:
                        ax.plot(epochs, values, label=f"{exp['name']} (valid)", 
                               color=colors[exp_idx], linestyle='-')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} Learning Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, exp_ids: List[str], save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive experiment report.
        
        Args:
            exp_ids: List of experiment IDs to include
            save_path: Optional path to save the report
        
        Returns:
            Report as markdown string
        """
        report = []
        report.append("# ASTE Experiment Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary table
        report.append("## Experiment Summary\n")
        df = self.compare_experiments(exp_ids)
        if not df.empty:
            # Select key columns for summary
            summary_cols = ['name', 'status', 'best_valid_f1', 'best_valid_precision', 'best_valid_recall']
            summary_cols = [col for col in summary_cols if col in df.columns]
            if summary_cols:
                report.append(df[summary_cols].to_markdown(index=False))
        
        # Detailed experiment information
        report.append("\n## Detailed Results\n")
        
        for exp_id in exp_ids:
            if exp_id not in self.experiments:
                continue
            
            exp = self.experiments[exp_id]
            report.append(f"### {exp['name']} ({exp_id})\n")
            
            # Configuration
            report.append("**Configuration:**")
            if 'config' in exp:
                for key, value in exp['config'].items():
                    report.append(f"- {key}: {value}")
            
            # Best metrics
            if 'final_metrics' in exp:
                report.append("\n**Final Test Metrics:**")
                for key, value in exp['final_metrics'].items():
                    if isinstance(value, float):
                        report.append(f"- {key}: {value:.4f}")
                    else:
                        report.append(f"- {key}: {value}")
            
            # Training history summary
            if 'valid' in exp.get('metrics', {}):
                valid_metrics = exp['metrics']['valid']
                if valid_metrics:
                    best_epoch = max(valid_metrics, key=lambda x: x.get('f1', 0))
                    report.append(f"\n**Best Validation (Epoch {best_epoch['epoch']}):**")
                    for key in ['f1', 'precision', 'recall']:
                        if key in best_epoch:
                            report.append(f"- {key}: {best_epoch[key]:.4f}")
            
            report.append("\n---\n")
        
        # Analysis and recommendations
        report.append("## Analysis\n")
        
        if not df.empty and 'best_valid_f1' in df.columns:
            best_exp_idx = df['best_valid_f1'].idxmax()
            best_exp = df.loc[best_exp_idx]
            report.append(f"**Best Performing Model:** {best_exp['name']}")
            report.append(f"- F1 Score: {best_exp['best_valid_f1']:.4f}")
            
            # Analyze what made it successful
            report.append("\n**Key Success Factors:**")
            config_cols = [col for col in df.columns if col.startswith('config_')]
            for col in config_cols:
                if pd.notna(best_exp[col]):
                    report.append(f"- {col.replace('config_', '')}: {best_exp[col]}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {save_path}")
        
        return report_text


class ModelComparator:
    """
    Compare different ASTE models systematically.
    """
    
    @staticmethod
    def compare_tagging_schemes(dataset, num_samples=5):
        """
        Compare JET^t and JET^o tagging on sample sentences.
        """
        from jeto_implementation import (
            triplets_to_tags_opinion_focused,
            analyze_dataset_for_jeto_advantages
        )
        from position_aware_tags_fixed import triplets_to_tags
        
        print("="*60)
        print("Tagging Scheme Comparison")
        print("="*60)
        
        # Analyze dataset characteristics
        stats = analyze_dataset_for_jeto_advantages(dataset)
        
        print(f"\nDataset Characteristics:")
        print(f"- Total sentences: {stats['total_sentences']}")
        print(f"- Single aspect → multiple opinions: {stats['single_aspect_multi_opinion']} cases")
        print(f"- Multiple aspects → single opinion: {stats['multi_aspect_single_opinion']} cases")
        
        # Show examples
        print(f"\n{num_samples} Example Comparisons:")
        print("-"*60)
        
        for i, item in enumerate(dataset[:num_samples]):
            sentence = item['sentence']
            triplets = item['triplets']
            
            if not triplets:
                continue
            
            print(f"\n{i+1}. Sentence: {sentence}")
            print(f"   Triplets: {len(triplets)}")
            
            # JET^t tags
            tags_t = triplets_to_tags(sentence, triplets)
            # JET^o tags
            tags_o = triplets_to_tags_opinion_focused(sentence, triplets)
            
            words = sentence.split()
            print("   JET^t tags:")
            for w, t in zip(words, tags_t):
                if t != 'O':
                    print(f"      {w:15} → {t}")
            
            print("   JET^o tags:")
            for w, t in zip(words, tags_o):
                if t != 'O':
                    print(f"      {w:15} → {t}")
    
    @staticmethod
    def error_analysis(predictions, gold_triplets, sentences):
        """
        Perform detailed error analysis.
        """
        errors = {
            'false_positives': [],
            'false_negatives': [],
            'boundary_errors': [],
            'sentiment_errors': []
        }
        
        for pred, gold, sent in zip(predictions, gold_triplets, sentences):
            # Implement detailed error categorization
            # This is simplified - you can expand it
            pred_keys = {(tuple(t['aspect_positions']), 
                         tuple(t['opinion_positions']), 
                         t['sentiment']) for t in pred}
            gold_keys = {(tuple(t['aspect_positions']), 
                         tuple(t['opinion_positions']), 
                         t['sentiment']) for t in gold}
            
            # False positives
            for p in pred_keys - gold_keys:
                errors['false_positives'].append({
                    'sentence': sent,
                    'prediction': p,
                    'gold': gold
                })
            
            # False negatives
            for g in gold_keys - pred_keys:
                errors['false_negatives'].append({
                    'sentence': sent,
                    'missed': g,
                    'predictions': pred
                })
        
        return errors


if __name__ == "__main__":
    # Example usage
    tracker = ExperimentTracker()
    
    # Start a JET^t experiment
    config_jett = {
        'model': 'JET_t',
        'learning_rate': 2e-5,
        'batch_size': 8,
        'max_epochs': 10,
        'dropout': 0.1,
        'max_offset': 10
    }
    
    exp_id_jett = tracker.start_experiment("JET_t_baseline", config_jett)
    
    # Simulate training
    for epoch in range(3):
        # Log training metrics
        tracker.log_metrics(exp_id_jett, epoch, {
            'loss': np.random.random() * 2,
            'f1': 0.5 + epoch * 0.1 + np.random.random() * 0.05,
            'precision': 0.6 + epoch * 0.08 + np.random.random() * 0.05,
            'recall': 0.4 + epoch * 0.12 + np.random.random() * 0.05
        }, phase='train')
        
        # Log validation metrics
        tracker.log_metrics(exp_id_jett, epoch, {
            'f1': 0.45 + epoch * 0.12 + np.random.random() * 0.05,
            'precision': 0.55 + epoch * 0.1 + np.random.random() * 0.05,
            'recall': 0.35 + epoch * 0.15 + np.random.random() * 0.05
        }, phase='valid')
    
    # Start a JET^o experiment
    config_jeto = {
        'model': 'JET_o',
        'learning_rate': 2e-5,
        'batch_size': 8,
        'max_epochs': 10,
        'dropout': 0.1,
        'max_offset': 10
    }
    
    exp_id_jeto = tracker.start_experiment("JET_o_opinion_focused", config_jeto)
    
    # Simulate training for JET^o
    for epoch in range(3):
        tracker.log_metrics(exp_id_jeto, epoch, {
            'loss': np.random.random() * 2,
            'f1': 0.48 + epoch * 0.11 + np.random.random() * 0.05,
            'precision': 0.58 + epoch * 0.09 + np.random.random() * 0.05,
            'recall': 0.38 + epoch * 0.13 + np.random.random() * 0.05
        }, phase='train')
        
        tracker.log_metrics(exp_id_jeto, epoch, {
            'f1': 0.43 + epoch * 0.13 + np.random.random() * 0.05,
            'precision': 0.53 + epoch * 0.11 + np.random.random() * 0.05,
            'recall': 0.33 + epoch * 0.16 + np.random.random() * 0.05
        }, phase='valid')
    
    # End experiments
    tracker.end_experiment(exp_id_jett, {'test_f1': 0.72, 'test_precision': 0.78, 'test_recall': 0.67})
    tracker.end_experiment(exp_id_jeto, {'test_f1': 0.74, 'test_precision': 0.76, 'test_recall': 0.71})
    
    # Compare experiments
    print("\n" + "="*60)
    print("Experiment Comparison")
    print("="*60)
    
    comparison_df = tracker.compare_experiments([exp_id_jett, exp_id_jeto])
    print(comparison_df[['name', 'status', 'final_test_f1', 'final_test_precision', 'final_test_recall']])
    
    # Generate report
    report = tracker.generate_report([exp_id_jett, exp_id_jeto], 
                                     save_path="experiments/experiment_report.md")
    
    # Plot learning curves
    fig = tracker.plot_learning_curves([exp_id_jett, exp_id_jeto])
    fig.savefig("experiments/learning_curves.png")
    print("\nLearning curves saved to: experiments/learning_curves.png")
    
    print("\nExperiment tracking demonstration complete!")