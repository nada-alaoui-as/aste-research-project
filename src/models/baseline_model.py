import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('src/data')
from load_aste import load_dataset
sys.path.append('src/models')
from position_aware_tags_fixed import triplets_to_tags, build_tag_vocab 
sys.path.append('src/evaluation')
from evaluate import evaluate_model  
from evaluate import predict_triplets

class ASTEDataset(Dataset):
    """Convert ASTE data to model input format"""
    def __init__(self, data, tag2idx, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.tag2idx = tag2idx
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
        
        tags = triplets_to_tags(sentence, item['triplets'])
        tag_ids = [self.tag2idx[tag] for tag in tags]
        tag_ids = tag_ids + [-100] * (self.max_len - len(tag_ids))

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            "tag_ids": torch.tensor(tag_ids),
            'sentence': sentence,
            'triplets': item['triplets']
        }

class SimpleASTEModel(nn.Module):
    """Simplified ASTE model"""
    def __init__(self, num_tags):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_tags)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

def calculate_loss(logits, tag_ids, attention_mask, tag2idx, device):
    # Create class weights - penalize O tag, boost rare tags
    num_tags = len(tag2idx)
    weights = torch.ones(num_tags)
    weights[tag2idx['O']] = 0.1  # O tag gets much lower weight
    
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=weights.to(device))
    loss = loss_fct(
        logits.view(-1, logits.shape[-1]),
        tag_ids.view(-1)
    )
    return loss

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for i, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tag_ids = batch['tag_ids'].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = calculate_loss(logits, tag_ids, attention_mask, tag2idx, device)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (i + 1) % 20 == 0:
            print(f"Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)

def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'tag_ids': torch.stack([item['tag_ids'] for item in batch]),
        'sentence': [item['sentence'] for item in batch],
        'triplets': [item['triplets'] for item in batch]
    }

if __name__ == "__main__":
    data_path = "data/raw/aste-data/ASTE-Data-V2-EMNLP2020/14res/train_triplets.txt"
    dataset = load_dataset(data_path)
    print(f"Loaded {len(dataset)} training examples")

    test_path = "data/raw/aste-data/ASTE-Data-V2-EMNLP2020/14res/test_triplets.txt"
    test_dataset = load_dataset(test_path)
    
    tag2idx, idx2tag = build_tag_vocab(dataset)
    num_tags = len(tag2idx)
    print(f"Vocabulary size: {num_tags}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    aste_dataset = ASTEDataset(dataset, tag2idx, tokenizer)
    dataloader = DataLoader(aste_dataset, batch_size=8, collate_fn=collate_fn)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleASTEModel(num_tags).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    
    print(f"\nDevice: {device}")
    print("Starting training...")

    for epoch in range(10):
        print(f"\n=== Epoch {epoch+1} ===")
        avg_loss = train_one_epoch(model, dataloader, optimizer, device)
        print(f"Average loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 2 == 0:
            metrics = evaluate_model(model, test_dataset[:50], tokenizer, tag2idx, idx2tag, device)
            print(f"Test F1: {metrics['f1']:.3f}")
    
    torch.save(model.state_dict(), 'experiments/baseline/model.pt')
    print("\nModel saved!")

    test_item = test_dataset[0]
    pred_triplets, pred_tags = predict_triplets(
        model, test_item['sentence'], tokenizer, tag2idx, idx2tag, device
    )

    print(f"\n=== Debug Output ===")
    print(f"Sentence: {test_item['sentence']}")
    print(f"Predicted tags: {pred_tags}")
    print(f"Predicted triplets: {pred_triplets}")
    print(f"Gold triplets: {test_item['triplets']}")
