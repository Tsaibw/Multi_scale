from multi_bert import multiBert
from scale import get_scaled_down_scores, separate_and_rescale_attributes_for_scoring
from evaluate import evaluation
from torch.utils.data import Dataset, DataLoader
import random
import torch
torch.manual_seed(11)

class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.file = file_path
        with open(file_path, 'rb') as f:
          self.data = pickle.load(f)
    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        document_single = self.data[0][idx]
        chunked_documents = [chunk[idx] for chunk in self.data[1]]
        label = self.data[2][idx]
        id_ = self.data[3][idx]
        length = [lengths[idx] for lengths in self.data[4]]
        return document_single, chunked_documents, label, id_, length


class NerConfig:
  def __init__(self):
    self.prompt = "p1"
    self.batch_size = 2
    self.data_sample_rate = 1.0
    self.r_dropout = 0.1
    self.device = "cuda"
    self.chunk_sizes = [90,30,130,10]
    self.warmup_proportion = 0.01
    self.result_file = "/content/drive/MyDrive/hw_ner/Multi_scale/Multi-Scale-BERT-AES/pred.txt"
    self.bert_model_path = "bert-base-uncased"
    self.train_file = "/content/drive/MyDrive/hw_ner/ASAP/dataset/Train_prompt/train_prompt_8.txt"
    self.test_file = "/content/drive/MyDrive/hw_ner/ASAP/dataset/Test_prompt/test_prompt_8.txt"
args = NerConfig()

# train normalize
import pickle
import os
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertConfig
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import clip_grad_value_
from tqdm import tqdm

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def print_gradients(model):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            print(f"{name} - Gradient Norm: {parameter.grad.norm().item()}")
        else:
            print(f"{name} - No gradient")


multi_bert_model = multiBert(args.chunk_sizes)  
multi_bert_model.to(args.device)  

optimizer = Adam(multi_bert_model.parameters(), lr=1e-5)  
criterion = torch.nn.MSELoss()  

num_epochs = 10

for i in range(1,9):
    train_dataset = CustomDataset(f"/home/tsaibw/Multi_scale/dataset/train/encode_prompt_{i}.pkl")
    eval_dataset = CustomDataset(f"/home/tsaibw/Multi_scale/dataset/test/encode_prompt_{i}.pkl")
    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=12, shuffle=False)
    train_loss , eval_loss = [] ,[] 
    os.makedirs(f"/home/tsaibw/Multi_scale/ckps/prompt{i}", exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"Model is running on: {args.device}")
        total_loss = 0
        eval_total_loss = 0
        for document_single, chunked_documents, label, id_, lengths in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
              document_single = document_single.to(args.device)
              optimizer.zero_grad()
            
              predictions = multi_bert_model(
                  document_single=document_single,
                  chunked_documents=chunked_documents,
                  device=args.device,
                  lengths=lengths
              )
            
              lla = label.tolist()
              ii = id_.to(torch.int).tolist()
              predictions_numpy = predictions.detach().cpu().numpy().reshape(-1)
              inverse_labels_batch = separate_and_rescale_attributes_for_scoring(lla, ii)
              inverse_predictions = separate_and_rescale_attributes_for_scoring(predictions_numpy, ii)
            
              # Calculate loss
              label = torch.tensor(label, dtype=torch.float32).to(args.device)
              loss = criterion(predictions.squeeze().float(), label)
              total_loss += loss.item()
              loss.backward()
              optimizer.step()
        print("prompt_id", ii)
        print("predictions:", predictions)
        print("train_predict: ",inverse_predictions)
        print("train_label: ",inverse_labels_batch)
        print("loss", loss)
        
        multi_bert_model.eval()
        eval_inverse_label = []
        eval_inverse_pred = []
        with torch.no_grad():
            for document_single, chunked_documents, label, id_, lengths in eval_loader:
                  document_single = document_single.to(args.device)
                
                  eval_predictions = multi_bert_model(
                      document_single=document_single,
                      chunked_documents=chunked_documents,
                      device=args.device,
                      lengths=lengths
                  )
                
                  # Calculate loss
                  loss = criterion(eval_predictions.squeeze(), torch.tensor(label, dtype=torch.float32).to(args.device))
                  eval_total_loss += loss.item()
                
                  id = id_.tolist()
                  label = label.tolist()
                  eval_pred = eval_predictions.detach().cpu().numpy().reshape(-1)
                  eval_inverse_pred.append(separate_and_rescale_attributes_for_scoring(eval_pred, id))
                  eval_inverse_label.append(separate_and_rescale_attributes_for_scoring(label, id))

        eval_inverse_pred_flattened = [item for sublist in eval_inverse_pred for item in sublist]
        eval_inverse_label_flattened = [item for sublist in eval_inverse_label for item in sublist]
        eval_inverse_pred = np.array(eval_inverse_pred_flattened).reshape(-1, 1)
        eval_inverse_label = np.array(eval_inverse_label).reshape(-1, 1)
        
        test_eva_res = evaluation(eval_inverse_label, eval_inverse_pred)
        pearson_score = float(test_eva_res[7])
        qwk_score = float(test_eva_res[8])
        print("pearson:", pearson_score)
        print("qwk:", qwk_score)
        
        
        average_loss = eval_total_loss / len(eval_loader)
        print(f"Test Loss: {average_loss}")
        print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")
        train_loss.append(total_loss / len(train_loader))
        eval_loss.append(average_loss)

        qwk_path = f"/home/tsaibw/Multi_scale/ckps/prompt{i}/prompt{i}.txt"
        with open(qwk_path, "a") as f:
            f.write(f"Epoch {epoch + 1}/{num_epochs}, QWK: {qwk_score}, Pearson: {pearson_score}, train_loss: {train_loss[-1]}, eval_loss: {eval_loss[-1]}\n")
  
        checkpoint_path = f"/home/tsaibw/Multi_scale/ckps/prompt{i}/epoch_{epoch+1}_checkpoint.pth.tar"
        save_checkpoint({
          'epoch': epoch + 1,
          'state_dict': multi_bert_model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'train_loss': train_loss,
          'eval_loss': eval_loss
        }, filename=checkpoint_path)
