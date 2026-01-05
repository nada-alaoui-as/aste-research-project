import torch
import sys
sys.path.append('src/data')
sys.path.append('src/models')
sys.path.append('src/evaluation')

from load_aste import load_dataset
from position_aware_tags import build_tag_vocab
from baseline_model import SimpleASTEModel
from evaluate import predict_triplets
from transformers import BertTokenizer

# Load data and model
test_path = "data/raw/aste-data/ASTE-Data-V2-EMNLP2020/14res/test_triplets.txt"
test_dataset = load_dataset(test_path)

train_path = "data/raw/aste-data/ASTE-Data-V2-EMNLP2020/14res/train_triplets.txt"
train_dataset = load_dataset(train_path)
tag2idx, idx2tag = build_tag_vocab(train_dataset)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleASTEModel(len(tag2idx)).to(device)
model.load_state_dict(torch.load('experiments/baseline/model.pt'))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Test on first 3 examples
for i in range(3):
    test_item = test_dataset[i]
    pred_triplets, pred_tags = predict_triplets(
        model, test_item['sentence'], tokenizer, tag2idx, idx2tag, device
    )
    
    print(f"\n=== Example {i+1} ===")
    print(f"Sentence: {test_item['sentence']}")
    print(f"Predicted tags: {pred_tags}")
    print(f"Predicted triplets: {pred_triplets}")
    print(f"Gold triplets: {test_item['triplets']}")