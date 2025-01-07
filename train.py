import os
import torch
import pickle
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from transformers import BertModel, BertConfig
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import clip_grad_value_
from tqdm import tqdm
from data.dataloader import CustomDataset
from model.multi_bert import multiBert
from data.scale import get_scaled_down_scores, separate_and_rescale_attributes_for_scoring
from utils.evaluate import evaluation

torch.manual_seed(11)

class NerConfig:
    def __init__(self):
        self.lr = 1e-5
        self.epoch = 15
        self.batch_size = 12
        self.device = "cuda"
        self.chunk_sizes = [90]
        self.data_file = "/home/tsaibw/Multi_scale/ckps/chunk_90"
        self.num_trait = 9
        self.alpha = 0.7
        self.delta = 0.7
        self.filter_num = 100
args = NerConfig()


# train normalize

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


def print_gradients(model):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            print(f"{name} - Gradient Norm: {parameter.grad.norm().item()}")
        else:
            print(f"{name} - No gradient")


for i in range(1,9):
    multi_bert_model = multiBert(args.chunk_sizes)  
    multi_bert_model.to(args.device)  
    optimizer = Adam(multi_bert_model.parameters(), lr = args.lr) 
    
    train_dataset = CustomDataset(f"/home/tsaibw/Multi_scale/dataset/train/encode_prompt_{i}.pkl")
    eval_dataset = CustomDataset(f"/home/tsaibw/Multi_scale/dataset/test/encode_prompt_{i}.pkl")
    test_dataset = CustomDataset(f"/home/tsaibw/Multi_scale/dataset/test/encode_prompt_{i}.pkl")
    dev_dataset = ConcatDataset([train_dataset, eval_dataset])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    train_loss_list , eval_loss_list = [] ,[] 
    os.makedirs(f"{args.data_file}/prompt{i}", exist_ok=True)
    
    for epoch in range(args.epoch):
        multi_bert_model.train()
        total_loss = 0
        evaluator = Evaluator(dev_dataset, test_dataset, 11)
        
        for document_single, chunked_documents, label, id_, lengths in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epoch}"):
            document_single = document_single.to(args.device)
            optimizer.zero_grad()
            
            loss, predict_score, scaled_score = multi_bert_model(
                    prompt_ids = ids
                    document_single = document_single,
                    chunked_documents = chunked_documents,
                    device = args.device,
                    lengths = lengths,
                    scaled_scores = label
            )
            
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        eval_loader = dev_loader + test_loader
        eval_loss, result = multi_bert_model.evaluate(eval_loader, test_loader, epoch, evaluator, device=args.device)
        
        print(f"Epoch {epoch}, Train Loss: {total_loss / len(train_loader)}")
        print(f"Test Loss: {eval_loss}")
        train_loss_list.append(total_loss / len(train_loader))
        eval_loss_list.append(eval_loss)

        qwk_path = f"{args.data_file}/prompt{i}/result.txt"
        with open(qwk_path, "a") as f:
            f.write(f"Epoch {epoch + 1}/{args.epoch}, result:{result}, train_loss: {train_loss_list[-1]}, eval_loss: {eval_loss_list[-1]}\n")
  
        checkpoint_path = f"{args.data_file}/prompt{i}/epoch_{epoch+1}_checkpoint.pth.tar"
        save_checkpoint({
          'epoch': epoch + 1,
          'state_dict': multi_bert_model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'train_loss': total_loss / len(train_loader),
          'eval_loss': eval_loss
        }, filename = checkpoint_path)
