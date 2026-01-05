"""
JET^o (Opinion-Focused) Implementation for ASTE
This module implements the opinion-focused variant of position-aware tagging
"""

import sys
sys.path.append('src/data')
from load_aste import load_dataset
from typing import List, Dict, Set, Tuple


def triplets_to_tags_opinion_focused(sentence: str, triplets: List[Dict], max_offset: int = 10) -> List[str]:
    """
    Convert triplets to opinion-focused position-aware tags (JET^o).
    
    Key difference from JET^t:
    - Tags opinion words (not aspect words)
    - Offsets (j, k) point from opinion to aspect
    
    Args:
        sentence: Input sentence string
        triplets: List of triplet dictionaries
        max_offset: Maximum allowed offset distance
    
    Returns:
        List of tags for each word in the sentence
    """
    if isinstance(triplets, dict):
        triplets = [triplets]
    
    words = sentence.split()
    tags = ['O'] * len(words)
    
    for triplet in triplets:
        aspect_pos = triplet['aspect_positions']
        opinion_pos = triplet['opinion_positions']
        sentiment = triplet['sentiment']
        
        # Calculate offsets FROM OPINION TO ASPECT (reversed from JET^t)
        j = aspect_pos[0] - opinion_pos[0]  # Start of aspect relative to opinion
        k = aspect_pos[-1] - opinion_pos[0]  # End of aspect relative to opinion
        
        # Clip offsets
        j = max(-max_offset, min(max_offset, j))
        k = max(-max_offset, min(max_offset, k))
        
        # Tag the OPINION span (not aspect)
        if len(opinion_pos) > 1:
            # Multi-word opinion
            tags[opinion_pos[0]] = f"B^{sentiment}_{j}_{k}"
            for idx in opinion_pos[1:-1]:
                tags[idx] = "I"
            tags[opinion_pos[-1]] = "E"
        else:
            # Single-word opinion
            tags[opinion_pos[0]] = f"S^{sentiment}_{j}_{k}"
    
    return tags


def build_tag_vocab_jeto(dataset: List[Dict]) -> Tuple[Dict, Dict]:
    """
    Build vocabulary for JET^o tagging scheme.
    
    Returns:
        tag2idx: Mapping from tags to indices
        idx2tag: Mapping from indices to tags
    """
    all_tags = set()
    
    # Collect all unique tags using opinion-focused tagging
    for item in dataset:
        tags = triplets_to_tags_opinion_focused(item['sentence'], item['triplets'])
        all_tags.update(tags)
    
    # Build mappings
    tag2idx = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    
    return tag2idx, idx2tag


def tags_to_triplets_opinion_focused(sentence: str, predicted_tags: List[str]) -> List[Dict]:
    """
    Convert predicted opinion-focused tags back to triplets.
    
    This is the inverse operation of triplets_to_tags_opinion_focused.
    
    Args:
        sentence: Input sentence
        predicted_tags: List of predicted tags (one per word)
    
    Returns:
        List of extracted triplets
    """
    words = sentence.split()
    triplets = []
    
    i = 0
    while i < len(predicted_tags):
        tag = predicted_tags[i]
        
        if tag.startswith("B") or tag.startswith("S"):
            # Parse the tag components
            parts = tag.split("_")
            if len(parts) != 3:
                i += 1
                continue
                
            sentiment_part = parts[0]
            j = int(parts[1])  # Offset to aspect start
            k = int(parts[2])  # Offset to aspect end
            
            # Extract sentiment
            sentiment = sentiment_part.split('^')[1] if '^' in sentiment_part else None
            if not sentiment:
                i += 1
                continue
            
            # Find opinion span
            opinion_positions = [i]
            
            if tag.startswith('B'):
                # Multi-word opinion
                for l in range(i+1, len(predicted_tags)):
                    if predicted_tags[l] in ['I', 'E']:
                        opinion_positions.append(l)
                        if predicted_tags[l] == 'E':
                            break
                    else:
                        break
            
            # Calculate aspect positions using offsets
            opinion_start = opinion_positions[0]
            aspect_start = opinion_start + j
            aspect_end = opinion_start + k
            
            # Validate aspect positions
            if aspect_start < 0 or aspect_end >= len(words) or aspect_start > aspect_end:
                i = opinion_positions[-1] + 1
                continue
            
            aspect_positions = list(range(aspect_start, aspect_end + 1))
            
            # Create triplet
            triplets.append({
                "aspect_positions": aspect_positions,
                "opinion_positions": opinion_positions,
                "sentiment": sentiment,
                "aspect": ' '.join([words[idx] for idx in aspect_positions]),
                "opinion": ' '.join([words[idx] for idx in opinion_positions])
            })
            
            i = opinion_positions[-1] + 1
        else:
            i += 1
    
    return triplets


def compare_tagging_schemes(sentence: str, triplets: List[Dict]):
    """
    Compare JET^t and JET^o tagging for the same input.
    Useful for understanding the differences.
    """
    from position_aware_tags_fixed import triplets_to_tags  # Import JET^t
    
    print(f"\nSentence: {sentence}")
    print(f"Triplets: {triplets}")
    print("\n" + "="*60)
    
    # JET^t tagging
    tags_jett = triplets_to_tags(sentence, triplets)
    print("JET^t tags (aspect-focused):")
    words = sentence.split()
    for word, tag in zip(words, tags_jett):
        if tag != 'O':
            print(f"  {word:15} -> {tag}")
    
    # JET^o tagging
    tags_jeto = triplets_to_tags_opinion_focused(sentence, triplets)
    print("\nJET^o tags (opinion-focused):")
    for word, tag in zip(words, tags_jeto):
        if tag != 'O':
            print(f"  {word:15} -> {tag}")
    
    # Reconstruct triplets from tags
    reconstructed = tags_to_triplets_opinion_focused(sentence, tags_jeto)
    print("\nReconstructed triplets from JET^o tags:")
    for t in reconstructed:
        print(f"  {t['aspect']} <- {t['opinion']} ({t['sentiment']})")


def analyze_dataset_for_jeto_advantages(dataset: List[Dict]) -> Dict:
    """
    Analyze dataset to identify cases where JET^o would be advantageous.
    
    Returns statistics about:
    - Single aspect with multiple opinions
    - Multiple aspects with single opinion
    - Distribution of aspect/opinion lengths
    """
    stats = {
        'total_sentences': len(dataset),
        'total_triplets': 0,
        'single_aspect_multi_opinion': 0,
        'multi_aspect_single_opinion': 0,
        'aspect_lengths': {},
        'opinion_lengths': {},
        'offset_distances': []
    }
    
    for item in dataset:
        triplets = item['triplets']
        stats['total_triplets'] += len(triplets)
        
        # Group by aspect positions
        aspect_to_opinions = {}
        opinion_to_aspects = {}
        
        for t in triplets:
            aspect_key = tuple(t['aspect_positions'])
            opinion_key = tuple(t['opinion_positions'])
            
            if aspect_key not in aspect_to_opinions:
                aspect_to_opinions[aspect_key] = []
            aspect_to_opinions[aspect_key].append(opinion_key)
            
            if opinion_key not in opinion_to_aspects:
                opinion_to_aspects[opinion_key] = []
            opinion_to_aspects[opinion_key].append(aspect_key)
            
            # Track lengths
            aspect_len = len(t['aspect_positions'])
            opinion_len = len(t['opinion_positions'])
            stats['aspect_lengths'][aspect_len] = stats['aspect_lengths'].get(aspect_len, 0) + 1
            stats['opinion_lengths'][opinion_len] = stats['opinion_lengths'].get(opinion_len, 0) + 1
            
            # Track offset distances
            distance = abs(t['opinion_positions'][0] - t['aspect_positions'][0])
            stats['offset_distances'].append(distance)
        
        # Count patterns
        for aspect, opinions in aspect_to_opinions.items():
            if len(set(opinions)) > 1:
                stats['single_aspect_multi_opinion'] += 1
        
        for opinion, aspects in opinion_to_aspects.items():
            if len(set(aspects)) > 1:
                stats['multi_aspect_single_opinion'] += 1
    
    # Calculate average offset distance
    if stats['offset_distances']:
        stats['avg_offset_distance'] = sum(stats['offset_distances']) / len(stats['offset_distances'])
        stats['max_offset_distance'] = max(stats['offset_distances'])
    
    return stats


if __name__ == "__main__":
    # Load dataset
    data_path = "data/raw/aste-data/ASTE-Data-V2-EMNLP2020/14res/train_triplets.txt"
    dataset = load_dataset(data_path)
    
    print("="*60)
    print("JET^o Implementation Test")
    print("="*60)
    
    # Test on example sentences
    test_examples = [
        {
            'sentence': "The food was delicious and fresh",
            'triplets': [
                {
                    'aspect': 'food',
                    'opinion': 'delicious',
                    'sentiment': 'POS',
                    'aspect_positions': [1],
                    'opinion_positions': [3]
                },
                {
                    'aspect': 'food',
                    'opinion': 'fresh',
                    'sentiment': 'POS',
                    'aspect_positions': [1],
                    'opinion_positions': [5]
                }
            ]
        }
    ]
    
    # This example shows JET^o's strength: one aspect, multiple opinions
    for example in test_examples:
        compare_tagging_schemes(example['sentence'], example['triplets'])
    
    # Build vocabulary
    print("\n" + "="*60)
    print("Building JET^o Vocabulary")
    print("="*60)
    tag2idx, idx2tag = build_tag_vocab_jeto(dataset)
    print(f"Total unique tags: {len(tag2idx)}")
    print(f"\nSample JET^o tags:")
    for tag in list(tag2idx.keys())[:10]:
        if tag not in ['O', 'I', 'E']:
            print(f"  {tag}")
    
    # Analyze dataset characteristics
    print("\n" + "="*60)
    print("Dataset Analysis for JET^o Advantages")
    print("="*60)
    stats = analyze_dataset_for_jeto_advantages(dataset[:100])  # Analyze first 100 for speed
    
    print(f"Total sentences analyzed: {stats['total_sentences']}")
    print(f"Total triplets: {stats['total_triplets']}")
    print(f"\nPattern Analysis:")
    print(f"  Single aspect → multiple opinions: {stats['single_aspect_multi_opinion']} cases")
    print(f"  Multiple aspects → single opinion: {stats['multi_aspect_single_opinion']} cases")
    print(f"\nLength Distribution:")
    print(f"  Aspect lengths: {dict(sorted(stats['aspect_lengths'].items()))}")
    print(f"  Opinion lengths: {dict(sorted(stats['opinion_lengths'].items()))}")
    print(f"\nOffset Statistics:")
    print(f"  Average distance: {stats.get('avg_offset_distance', 0):.2f} words")
    print(f"  Maximum distance: {stats.get('max_offset_distance', 0)} words")