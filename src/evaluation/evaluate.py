"""
Evaluation metrics for ASTE
"""
import torch
from typing import List, Dict

def tags_to_triplets(sentence: str, predicted_tags: List[str]) -> List[Dict]:
    """Convert predicted tag sequence back to triplets."""
    triplets = []
    for i, tag in enumerate(predicted_tags):
        if tag.startswith("B") or tag.startswith("S"):
            parts = tag.split("_")
            sentiment_part = parts[0]
            j = int(parts[1])
            k = int(parts[2])
            
            sentiment = sentiment_part.split('^')[1]
            aspect_positions = [i]
            
            if tag.startswith('B'):
                for l in range(i+1, len(predicted_tags)):
                    if predicted_tags[l] in ['I', 'E']:
                        aspect_positions.append(l)
                        if predicted_tags[l] == 'E':
                            break
            
            aspect_start = aspect_positions[0]
            opinion_start = aspect_start + j
            opinion_end = aspect_start + k
            opinion_positions = list(range(opinion_start, opinion_end + 1))

            triplets.append({
                "aspect_positions": aspect_positions,
                "opinion_positions": opinion_positions,
                "sentiment": sentiment
            })
    return triplets


def calculate_f1(pred_triplets: List, gold_triplets: List) -> Dict:
    """
    Calculate F1 with relaxed matching:
    - Aspect positions must match exactly
    - Opinion positions need >= 50% overlap
    - Sentiment must match
    """
    true_pos = 0
    false_pos = 0
    false_neg = 0
    
    matched_gold = set()
    
    for pred in pred_triplets:
        best_match = None
        best_overlap = 0
        
        for i, gold in enumerate(gold_triplets):
            if i in matched_gold:
                continue
                
            # Check aspect match
            if pred['aspect_positions'] != gold['aspect_positions']:
                continue
            
            # Check sentiment match
            if pred['sentiment'] != gold['sentiment']:
                continue
            
            # Check opinion overlap
            pred_opinion = set(pred['opinion_positions'])
            gold_opinion = set(gold['opinion_positions'])
            
            overlap = len(pred_opinion & gold_opinion)
            union = len(pred_opinion | gold_opinion)
            
            iou = overlap / union if union > 0 else 0
            
            if iou > best_overlap:
                best_overlap = iou
                best_match = i
        
        if best_overlap >= 0.5:  # At least 50% overlap
            true_pos += 1
            matched_gold.add(best_match)
        else:
            false_pos += 1
    
    false_neg = len(gold_triplets) - len(matched_gold)
    
    precision = true_pos/(true_pos+false_pos) if (true_pos+false_pos) > 0 else 0.0
    recall = true_pos/(true_pos+false_neg) if (true_pos+false_neg) > 0 else 0.0
    f1 = 2*(precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_pos,
        'false_positives': false_pos,
        'false_negatives': false_neg
    }

def predict_triplets(model, sentence, tokenizer, tag2idx, idx2tag, device):
    """Given a sentence, predict its triplets."""
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
        logits = model(input_ids, attention_mask)
    
    predicted_tag_ids = torch.argmax(logits, dim=-1).squeeze().cpu().tolist()
    
    predicted_tags = []
    for i in range(num_words):
        tag_id = predicted_tag_ids[i]
        tag = idx2tag.get(tag_id, 'O')
        predicted_tags.append(tag)
    
    triplets = tags_to_triplets(sentence, predicted_tags)
    return triplets, predicted_tags

def evaluate_model(model, dataset, tokenizer, tag2idx, idx2tag, device):
    """Evaluate on entire dataset"""
    all_preds = []
    all_golds = []
    
    for item in dataset:
        pred_triplets, _ = predict_triplets(
            model, item['sentence'], tokenizer, tag2idx, idx2tag, device
        )
        all_preds.extend(pred_triplets)
        all_golds.extend(item['triplets'])
    
    metrics = calculate_f1(all_preds, all_golds)
    return metrics