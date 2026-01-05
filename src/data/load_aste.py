"""
ASTE Data Loader
Utilities for loading and parsing Aspect Sentiment Triplet Extraction datasets.
"""

import ast
from typing import List, Dict, Tuple


def parse_line_complete(line: str) -> Tuple[str, List[Dict]]:
    """
    Parse a single line from ASTE dataset file.
    
    Format: "sentence####[([aspect_pos], [opinion_pos], 'sentiment'), ...]"
    
    Args:
        line: Raw line from dataset file
        
    Returns:
        sentence: The review sentence
        result: List of triplet dictionaries containing aspect, opinion, sentiment info
        
    Example:
        Input: "The food was great####[([1], [3], 'POS')]"
        Output: ("The food was great", 
                [{'aspect': 'food', 'opinion': 'great', 'sentiment': 'POS', ...}])
    """
    # Split sentence and annotation
    parts = line.split("####")
    sentence = parts[0]
    triplets = parts[1]
    
    # Parse triplets from string format to Python list
    triplets = ast.literal_eval(triplets)
    words = sentence.split()
    
    result = []
    for triplet in triplets:
        aspect_positions = triplet[0]
        opinion_positions = triplet[1]
        sentiment = triplet[2]
        
        # Extract actual words using positions
        aspect_words = [words[i] for i in aspect_positions]
        opinion_words = [words[i] for i in opinion_positions]
        
        aspect_text = ' '.join(aspect_words)
        opinion_text = ' '.join(opinion_words)
        
        result.append({
            'aspect': aspect_text,
            'opinion': opinion_text,
            'sentiment': sentiment,
            'aspect_positions': aspect_positions,
            'opinion_positions': opinion_positions
        })
    
    return sentence, result


def load_dataset(file_path: str) -> List[Dict]:
    """
    Load entire ASTE dataset file.
    
    Args:
        file_path: Path to dataset file (e.g., train_triplets.txt)
        
    Returns:
        List of dictionaries with 'sentence' and 'triplets' keys
    """
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                sentence, triplets = parse_line_complete(line)
                data.append({
                    'sentence': sentence,
                    'triplets': triplets
                })
    
    return data


def main():
    """Demo usage of data loader"""
    data_path = "data/raw/aste-data/ASTE-Data-V2-EMNLP2020/14res/train_triplets.txt"
    dataset = load_dataset(data_path)
    
    print("\n=== Dataset Statistics ===")
    print(f"Total sentences: {len(dataset)}")
    
    total_triplets = sum(len(item['triplets']) for item in dataset)
    print(f"Total triplets: {total_triplets}")
    
    # Show first 3 examples
    print("\n=== First 3 Examples ===")
    for i in range(min(3, len(dataset))):
        item = dataset[i]
        print(f"\nSentence {i+1}: {item['sentence'][:80]}...")
        for j, t in enumerate(item['triplets']):
            print(f"  Triplet {j+1}: [{t['aspect']}] -> [{t['opinion']}] ({t['sentiment']})")


if __name__ == "__main__":
    main()