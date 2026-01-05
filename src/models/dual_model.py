"""
Enhanced ASTE Model with JET^o Support
Implements both JET^t and JET^o variants with ensemble capabilities
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import sys
import numpy as np
from typing import List, Dict, Tuple, Set

sys.path.append('src')
from data.load_aste import load_dataset
from models.position_aware_tags_fixed import triplets_to_tags, build_tag_vocab
from models.jeto_implementation import (
    triplets_to_tags_opinion_focused, 
    build_tag_vocab_jeto,
    tags_to_triplets_opinion_focused
)
from evaluation.evaluate import evaluate_model, calculate_f1, tags_to_triplets


class ASTEDatasetDual(Dataset):
    """Dataset that supports both JET^t and JET^o tagging schemes"""
    
    def __init__(self, data, tag2idx_jett, tag2idx_jeto, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.tag2idx_jett = tag2idx_jett
        self.tag2idx_jeto = tag2idx_jeto
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item['sentence']
        
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Generate tags for both schemes
        tags_jett = triplets_to_tags(sentence, item['triplets'])
        tags_jeto = triplets_to_tags_opinion_focused(sentence, item['triplets'])
        
        # Convert to IDs
        tag_ids_jett = [self.tag2idx_jett.get(tag, self.tag2idx_jett['O']) for tag in tags_jett]
        tag_ids_jeto = [self.tag2idx_jeto.get(tag, self.tag2idx_jeto['O']) for tag in tags_jeto]
        
        # Pad
        tag_ids_jett = tag_ids_jett + [-100] * (self.max_len - len(tag_ids_jett))
        tag_ids_jeto = tag_ids_jeto + [-100] * (self.max_len - len(tag_ids_jeto))
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'tag_ids_jett': torch.tensor(tag_ids_jett),
            'tag_ids_jeto': torch.tensor(tag_ids_jeto),
            'sentence': sentence,
            'triplets': item['triplets']
        }


class DualASTEModel(nn.Module):
    """
    Enhanced ASTE model supporting both JET^t and JET^o.
    Can be used in three modes:
    1. 'jett' - Only use JET^t branch
    2. 'jeto' - Only use JET^o branch
    3. 'ensemble' - Use both branches
    """
    
    def __init__(self, num_tags_jett, num_tags_jeto, mode='ensemble'):
        super().__init__()
        self.mode = mode
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        
        # Separate classifiers for each tagging scheme
        self.classifier_jett = nn.Linear(768, num_tags_jett) if mode in ['jett', 'ensemble'] else None
        self.classifier_jeto = nn.Linear(768, num_tags_jeto) if mode in ['jeto', 'ensemble'] else None
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        results = {}
        if self.classifier_jett is not None:
            results['logits_jett'] = self.classifier_jett(sequence_output)
        if self.classifier_jeto is not None:
            results['logits_jeto'] = self.classifier_jeto(sequence_output)
        
        return results


def calculate_dual_loss(outputs, tag_ids_jett, tag_ids_jeto, tag2idx_jett, tag2idx_jeto, device, mode='ensemble'):
    """Calculate loss for dual model"""
    total_loss = 0
    num_losses = 0
    
    if 'logits_jett' in outputs and tag_ids_jett is not None:
        # JET^t loss with class weights
        num_tags = len(tag2idx_jett)
        weights = torch.ones(num_tags)
        weights[tag2idx_jett['O']] = 0.1
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=weights.to(device))
        loss_jett = loss_fct(
            outputs['logits_jett'].view(-1, outputs['logits_jett'].shape[-1]),
            tag_ids_jett.view(-1)
        )
        total_loss += loss_jett
        num_losses += 1
    
    if 'logits_jeto' in outputs and tag_ids_jeto is not None:
        # JET^o loss with class weights
        num_tags = len(tag2idx_jeto)
        weights = torch.ones(num_tags)
        weights[tag2idx_jeto['O']] = 0.1
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=weights.to(device))
        loss_jeto = loss_fct(
            outputs['logits_jeto'].view(-1, outputs['logits_jeto'].shape[-1]),
            tag_ids_jeto.view(-1)
        )
        total_loss += loss_jeto
        num_losses += 1
    
    return total_loss / max(num_losses, 1)


def ensemble_predictions(pred_jett: List[Dict], pred_jeto: List[Dict], strategy='union') -> List[Dict]:
    """
    Combine predictions from JET^t and JET^o.
    
    Strategies:
    - 'union': Take all unique triplets from both
    - 'intersection': Only triplets found by both
    - 'jett_priority': Start with JET^t, add non-overlapping from JET^o
    - 'jeto_priority': Start with JET^o, add non-overlapping from JET^t
    """
    
    def triplet_key(t):
        """Create a hashable key for triplet comparison"""
        return (tuple(t['aspect_positions']), 
                tuple(t['opinion_positions']), 
                t['sentiment'])
    
    def triplets_overlap(t1, t2):
        """Check if two triplets have overlapping spans"""
        aspect_overlap = set(t1['aspect_positions']) & set(t2['aspect_positions'])
        opinion_overlap = set(t1['opinion_positions']) & set(t2['opinion_positions'])
        return len(aspect_overlap) > 0 and len(opinion_overlap) > 0
    
    if strategy == 'union':
        # Combine all unique triplets
        seen = set()
        result = []
        for t in pred_jett + pred_jeto:
            key = triplet_key(t)
            if key not in seen:
                seen.add(key)
                result.append(t)
        return result
    
    elif strategy == 'intersection':
        # Only triplets found by both models
        jett_keys = {triplet_key(t) for t in pred_jett}
        result = []
        for t in pred_jeto:
            if triplet_key(t) in jett_keys:
                result.append(t)
        return result
    
    elif strategy == 'jett_priority':
        # Start with JET^t predictions
        result = list(pred_jett)
        # Add non-overlapping JET^o predictions
        for t_jeto in pred_jeto:
            overlap = False
            for t_jett in pred_jett:
                if triplets_overlap(t_jeto, t_jett):
                    overlap = True
                    break
            if not overlap:
                result.append(t_jeto)
        return result
    
    elif strategy == 'jeto_priority':
        # Start with JET^o predictions
        result = list(pred_jeto)
        # Add non-overlapping JET^t predictions
        for t_jett in pred_jett:
            overlap = False
            for t_jeto in pred_jeto:
                if triplets_overlap(t_jett, t_jeto):
                    overlap = True
                    break
            if not overlap:
                result.append(t_jett)
        return result
    
    else:
        raise ValueError(f"Unknown ensemble strategy: {strategy}")


def predict_ensemble(model, sentence, tokenizer, tag2idx_jett, idx2tag_jett, 
                     tag2idx_jeto, idx2tag_jeto, device, ensemble_strategy='jeto_priority'):
    """Make predictions using ensemble model"""
    model.eval()
    words = sentence.split()
    num_words = len(words)
    
    encoding = tokenizer(
        sentence, max_length=128, padding='max_length',
        truncation=True, return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    predictions = []
    
    # Get JET^t predictions
    if 'logits_jett' in outputs:
        logits_jett = outputs['logits_jett']
        predicted_tag_ids = torch.argmax(logits_jett, dim=-1).squeeze().cpu().tolist()
        predicted_tags_jett = []
        for i in range(num_words):
            tag_id = predicted_tag_ids[i]
            tag = idx2tag_jett.get(tag_id, 'O')
            predicted_tags_jett.append(tag)
        triplets_jett = tags_to_triplets(sentence, predicted_tags_jett)
    else:
        triplets_jett = []
    
    # Get JET^o predictions
    if 'logits_jeto' in outputs:
        logits_jeto = outputs['logits_jeto']
        predicted_tag_ids = torch.argmax(logits_jeto, dim=-1).squeeze().cpu().tolist()
        predicted_tags_jeto = []
        for i in range(num_words):
            tag_id = predicted_tag_ids[i]
            tag = idx2tag_jeto.get(tag_id, 'O')
            predicted_tags_jeto.append(tag)
        triplets_jeto = tags_to_triplets_opinion_focused(sentence, predicted_tags_jeto)
    else:
        triplets_jeto = []
    
    # Combine predictions
    if model.mode == 'ensemble':
        final_triplets = ensemble_predictions(triplets_jett, triplets_jeto, ensemble_strategy)
    elif model.mode == 'jett':
        final_triplets = triplets_jett
    else:  # jeto
        final_triplets = triplets_jeto
    
    return final_triplets, triplets_jett, triplets_jeto


def train_dual_model(model, dataloader, optimizer, device, tag2idx_jett, tag2idx_jeto, mode='ensemble'):
    """Train the dual model"""
    model.train()
    total_loss = 0
    
    for i, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tag_ids_jett = batch['tag_ids_jett'].to(device) if mode in ['jett', 'ensemble'] else None
        tag_ids_jeto = batch['tag_ids_jeto'].to(device) if mode in ['jeto', 'ensemble'] else None
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        
        loss = calculate_dual_loss(
            outputs, tag_ids_jett, tag_ids_jeto, 
            tag2idx_jett, tag2idx_jeto, device, mode
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (i + 1) % 20 == 0:
            print(f"Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def collate_fn_dual(batch):
    """Collate function for dual dataset"""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'tag_ids_jett': torch.stack([item['tag_ids_jett'] for item in batch]),
        'tag_ids_jeto': torch.stack([item['tag_ids_jeto'] for item in batch]),
        'sentence': [item['sentence'] for item in batch],
        'triplets': [item['triplets'] for item in batch]
    }


if __name__ == "__main__":
    # Load data
    data_path = "data/raw/aste-data/ASTE-Data-V2-EMNLP2020/14res/train_triplets.txt"
    dataset = load_dataset(data_path)[:500]  # Use subset for faster testing
    print(f"Loaded {len(dataset)} training examples")
    
    test_path = "data/raw/aste-data/ASTE-Data-V2-EMNLP2020/14res/test_triplets.txt"
    test_dataset = load_dataset(test_path)[:50]
    
    # Build vocabularies for both schemes
    print("\nBuilding vocabularies...")
    tag2idx_jett, idx2tag_jett = build_tag_vocab(dataset)
    tag2idx_jeto, idx2tag_jeto = build_tag_vocab_jeto(dataset)
    
    print(f"JET^t vocabulary size: {len(tag2idx_jett)}")
    print(f"JET^o vocabulary size: {len(tag2idx_jeto)}")
    
    # Create dataset and dataloader
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dual_dataset = ASTEDatasetDual(dataset, tag2idx_jett, tag2idx_jeto, tokenizer)
    dataloader = DataLoader(dual_dataset, batch_size=4, collate_fn=collate_fn_dual)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualASTEModel(len(tag2idx_jett), len(tag2idx_jeto), mode='ensemble').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    
    print(f"\nDevice: {device}")
    print(f"Model mode: {model.mode}")
    print("Starting training...")
    
    # Training loop
    for epoch in range(10):  # Fewer epochs for testing
        print(f"\n=== Epoch {epoch+1} ===")
        avg_loss = train_dual_model(model, dataloader, optimizer, device, tag2idx_jett, tag2idx_jeto, model.mode)
        print(f"Average loss: {avg_loss:.4f}")
        
        # Evaluation
        if (epoch + 1) % 1 == 0:
            model.eval()
            all_preds = []
            all_golds = []
            
            for test_item in test_dataset[:10]:  # Quick evaluation on subset
                pred_final, pred_jett, pred_jeto = predict_ensemble(
                    model, test_item['sentence'], tokenizer,
                    tag2idx_jett, idx2tag_jett, tag2idx_jeto, idx2tag_jeto,
                    device, ensemble_strategy='jeto_priority'
                )
                all_preds.extend(pred_final)
                all_golds.extend(test_item['triplets'])
            
            metrics = calculate_f1(all_preds, all_golds)
            print(f"Ensemble F1: {metrics['f1']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'tag2idx_jett': tag2idx_jett,
        'tag2idx_jeto': tag2idx_jeto,
        'idx2tag_jett': idx2tag_jett,
        'idx2tag_jeto': idx2tag_jeto,
        'mode': model.mode
    }, 'experiments/baseline/dual_model.pt')
    print("\nDual model saved!")
    
    # Demonstrate ensemble advantages
    print("\n" + "="*60)
    print("Demonstrating Ensemble Advantages")
    print("="*60)
    
    test_item = test_dataset[0]
    model.eval()
    
    pred_final, pred_jett, pred_jeto = predict_ensemble(
        model, test_item['sentence'], tokenizer,
        tag2idx_jett, idx2tag_jett, tag2idx_jeto, idx2tag_jeto,
        device, ensemble_strategy='jeto_priority'
    )
    s=test_item["sentence"].split()
    print(f"Sentence: {test_item['sentence']}")
    print(f"\nGold triplets ({len(test_item['triplets'])}):")
    for t in test_item['triplets']:
        print(f"  {s[t['aspect_positions'][0]:t['aspect_positions'][-1]+1]} ← {s[t['opinion_positions'][0]:t['opinion_positions'][-1]+1]} ({t['sentiment']})")
    
    print(f"\nJET^t predictions ({len(pred_jett)}):")
    for t in pred_jett:
        print(f"  {s[t['aspect_positions'][0]:t['aspect_positions'][-1]+1]} ← {s[t['opinion_positions'][0]:t['opinion_positions'][-1]+1]} ({t['sentiment']})")
    
    print(f"\nJET^o predictions ({len(pred_jeto)}):")
    for t in pred_jeto:
        print(f"  {s[t['aspect_positions'][0]:t['aspect_positions'][-1]+1]} ← {s[t['opinion_positions'][0]:t['opinion_positions'][-1]+1]} ({t['sentiment']})")
    
    print(f"\nEnsemble predictions ({len(pred_final)}):")
    for t in pred_final:
        print(f"  {s[t['aspect_positions'][0]:t['aspect_positions'][-1]+1]} ← {s[t['opinion_positions'][0]:t['opinion_positions'][-1]+1]} ({t['sentiment']})")