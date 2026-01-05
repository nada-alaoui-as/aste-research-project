import sys
sys.path.append('src/data')
from load_aste import load_dataset

def triplets_to_tags(sentence, triplets, max_offset=10):
    """Convert triplets to position-aware tags"""
    if isinstance(triplets, dict):
        triplets = [triplets]
    
    words = sentence.split()
    tags = ['O'] * len(words)
    
    for triplet in triplets:
        aspect_pos = triplet['aspect_positions']
        opinion_pos = triplet['opinion_positions']
        sentiment = triplet['sentiment']
        
        # Calculate offsets
        j = opinion_pos[0] - aspect_pos[0]
        k = opinion_pos[-1] - aspect_pos[0]
        
        # Clip offsets
        j = max(-max_offset, min(max_offset, j))
        k = max(-max_offset, min(max_offset, k))
        
        # Multi-word aspect
        if len(aspect_pos) > 1:
            tags[aspect_pos[0]] = f"B^{sentiment}_{j}_{k}"
            for idx in aspect_pos[1:-1]:
                tags[idx] = "I"
            tags[aspect_pos[-1]] = "E"
        else:  # Single-word aspect
            tags[aspect_pos[0]] = f"S^{sentiment}_{j}_{k}"
    
    return tags


def build_tag_vocab(dataset):
    """Build vocabulary from dataset - FIXED VERSION"""
    all_tags = set()
    
    # Collect all unique tags
    for item in dataset:
        tags = triplets_to_tags(item['sentence'], item['triplets'])
        all_tags.update(tags)
    
    # Build mappings AFTER collecting all tags
    tag2idx = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    
    return tag2idx, idx2tag


if __name__ == "__main__":
    # Test the fix
    data_path = "data/raw/aste-data/ASTE-Data-V2-EMNLP2020/14res/train_triplets.txt"
    dataset = load_dataset(data_path)
    
    tag2idx, idx2tag = build_tag_vocab(dataset)
    print(f"Total unique tags: {len(tag2idx)}")
    print(f"First 20 tags: {list(tag2idx.keys())[:20]}")
    
    # Verify it's actually capturing all tags
    print(f"\nTag 'O' index: {tag2idx.get('O', 'NOT FOUND')}")
    print(f"Sample position-aware tags:")
    for tag in list(tag2idx.keys())[:5]:
        if tag != 'O' and tag != 'I' and tag != 'E':
            print(f"  {tag}")