
#一定要先，不然torch會偵測不到
!export CUDA_VISIBLE_DEVICES=4
%set_env CUDA_VISIBLE_DEVICES=4


import os
import torch
import pickle
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    train_loss_list , eval_loss_list = [] ,[] 
    os.makedirs(f"{args.data_file}/prompt{i}", exist_ok=True)
    
    for epoch in range(args.epoch):
        multi_bert_model.train()
        total_loss = 0

        for document_single, chunked_documents, label, id_, lengths in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epoch}"):
            document_single = document_single.to(args.device)
            optimizer.zero_grad()
            
            predictions = multi_bert_model(
                    document_single=document_single,
                    chunked_documents=chunked_documents,
                    device=args.device,
                    lengths=lengths
            )
            
            loss, inverse_predictions, inverse_labels = multi_bert_model.compute_loss(predictions, label, id_, args.device)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
        
        eval_loss, qwk_score, pearson_score = multi_bert_model.evaluate(eval_loader, device = args.device)
        
        print(f"Epoch {epoch}, Train Loss: {total_loss / len(train_loader)}")
        print(f"Test Loss: {eval_loss}")
        train_loss_list.append(total_loss / len(train_loader))
        eval_loss_list.append(eval_loss)

        qwk_path = f"{args.data_file}/prompt{i}/result.txt"
        with open(qwk_path, "a") as f:
            f.write(f"Epoch {epoch + 1}/{args.epoch}, QWK: {qwk_score}, Pearson: {pearson_score}, train_loss: {train_loss_list[-1]}, eval_loss: {eval_loss_list[-1]}\n")
  
        checkpoint_path = f"{args.data_file}/prompt{i}/epoch_{epoch+1}_checkpoint.pth.tar"
        save_checkpoint({
          'epoch': epoch + 1,
          'state_dict': multi_bert_model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'train_loss': total_loss / len(train_loader),
          'eval_loss': eval_loss
        }, filename = checkpoint_path)
