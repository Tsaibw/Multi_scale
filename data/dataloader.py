from torch.utils.data import Dataset, DataLoader
import pickle

class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.file = file_path
        with open(file_path, 'rb') as f:
          self.data = pickle.load(f)
    def __len__(self):
        return int(len(self.data[0]))

    def __getitem__(self, idx):
        document_single = self.data[0][idx]
        chunked_documents = [chunk[idx] for chunk in self.data[1]]
        label = self.data[2][idx]
        id_ = self.data[3][idx]
        length = [lengths[idx] for lengths in self.data[4]]
        
        return document_single, chunked_documents, label, id_, length
