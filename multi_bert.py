#document_bert_architectures
import torch
from torch import nn
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertPreTrainedModel, BertConfig, BertModel
from scale import get_scaled_down_scores, separate_and_rescale_attributes_for_scoring
from evaluate import evaluation
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(7)


class DocumentBertSentenceChunkAttentionLSTM(BertPreTrainedModel):
    def __init__(self):
        bert_model_config = BertConfig.from_pretrained('bert-base-uncased')
        super(DocumentBertSentenceChunkAttentionLSTM, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        for layer in self.bert.encoder.layer[:11]:
            for param in layer.parameters():
                param.requires_grad = False
        # for param in self.bert.parameters():#212121
        #     param.requires_grad = False
        
        self.dropout = nn.Dropout(0.1)
        self.lstm = LSTM(bert_model_config.hidden_size,bert_model_config.hidden_size, batch_first = True)
        self.mlp = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(bert_model_config.hidden_size, 1),
            nn.Sigmoid()#
        )
        self.w_omega = nn.Parameter(torch.Tensor(bert_model_config.hidden_size, bert_model_config.hidden_size))
        self.b_omega = nn.Parameter(torch.Tensor(1, bert_model_config.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(bert_model_config.hidden_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        nn.init.uniform_(self.b_omega, -0.1, 0.1)
        self.mlp.apply(init_weights)


    def forward(self, document_batch: torch.Tensor, device="cuda", bert_batch_size=0, length=0):
        bert_output = torch.zeros(size=(document_batch.shape[0],
                      # min(document_batch.shape[1],
                      # bert_batch_size),
                      document_batch.shape[1],
                      self.bert.config.hidden_size), dtype=torch.float, device=device)
        for doc_id in range(document_batch.shape[0]):
            bert_output_temp = self.bert(document_batch[doc_id][:length[doc_id], 0],
                            token_type_ids=document_batch[doc_id][:length[doc_id], 1],
                            attention_mask=document_batch[doc_id][:length[doc_id], 2])[1]

            bert_output_temp = torch.nan_to_num(bert_output_temp, nan=0.0)
            bert_output[doc_id][:length[doc_id]] = self.dropout(bert_output_temp)

        packed_input = pack_padded_sequence(bert_output, length, batch_first=True, enforce_sorted=False)
        packed_output, (_, _) = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # print("output.shape", output.shape)
        # print(output)
        # (batch_size, seq_len, num_hiddens) 768
        attention_w = torch.tanh(torch.matmul(output, self.w_omega) + self.b_omega)
        attention_u = torch.matmul(attention_w, self.u_omega)  # (batch_size, seq_len, 1)
        attention_score = F.softmax(attention_u, dim=1)  # (batch_size, seq_len, 1)
        attention_hidden = output * attention_score  # (batch_size, seq_len, num_hiddens)
        attention_hidden = torch.sum(attention_hidden, dim=1)  # 加權求和 (batch_size, num_hiddens)
        # print("attention_hidden", attention_hidden.shape, attention_hidden) #(batcg_sizem, 768)
        prediction = self.mlp(attention_hidden)
        assert prediction.shape[0] == document_batch.shape[0]
        # print("predictionpredictionpredictionprediction:", prediction)
        return prediction


class DocumentBertCombineWordDocumentLinear(BertPreTrainedModel):
    def __init__(self):
        bert_model_config = BertConfig.from_pretrained('bert-base-uncased')
        super(DocumentBertCombineWordDocumentLinear, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        for layer in self.bert.encoder.layer[:11]:
            for param in layer.parameters():
                param.requires_grad = False
        # for param in self.bert.parameters():
        #     param.requires_grad = False
    
        self.bert_batch_size = 1
        self.dropout = nn.Dropout(0.1)
        self.mlp = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(bert_model_config.hidden_size * 2, 1),
            nn.Sigmoid()#
        )
        self.mlp.apply(init_weights)

    def forward(self, document_batch: torch.Tensor, device="cuda"):
        bert_output = torch.zeros(size=(document_batch.shape[0], # batch_size
                      min(document_batch.shape[1], self.bert_batch_size), #分段的長度
                      self.bert.config.hidden_size * 2),
                      dtype=torch.float, device="cuda")
        for doc_id in range(document_batch.shape[0]):
            # 只取分段一
            all_bert_output_info = self.bert(document_batch[doc_id][:self.bert_batch_size,0],
                              token_type_ids=document_batch[doc_id][:self.bert_batch_size, 1],
                              attention_mask=document_batch[doc_id][:self.bert_batch_size, 2])
            bert_token_max = torch.max(all_bert_output_info[0], 1)
            bert_output[doc_id][:self.bert_batch_size] = torch.cat((bert_token_max.values, all_bert_output_info[1]), 1)

        prediction = self.mlp(bert_output.view(bert_output.shape[0], -1))
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction


class multiBert(nn.Module):

    def __init__(self, chunk_sizes):
        super(multiBert, self).__init__()
        self.chunk = DocumentBertSentenceChunkAttentionLSTM()
        self.linear = DocumentBertCombineWordDocumentLinear()
        self.chunk_sizes = chunk_sizes
        self.mse_loss = nn.MSELoss()
        
    def forward( self, document_single: torch.Tensor, chunked_documents: torch.Tensor, device="cuda", lengths=0):
        prediction_single = self.linear(document_single, device = device) #(batch_size, 1)
        prediction_chunked = torch.zeros_like(prediction_single)

        for chunk_index in range(len(self.chunk_sizes)):
            batch_document_tensor_chunk = chunked_documents[chunk_index].to(device)
            length = lengths[chunk_index]
            predictions_chunk = self.chunk(batch_document_tensor_chunk, device = device, length = length)
            prediction_chunked += predictions_chunk

        batch_predictions_word_chunk_sentence_doc = torch.add(prediction_single, prediction_chunked)
        return batch_predictions_word_chunk_sentence_doc


    def compute_loss(self, predictions, labels, ids):
        predictions_numpy = predictions.detach().cpu().numpy().reshape(-1)
        inverse_predictions = separate_and_rescale_attributes_for_scoring(predictions_numpy, ids.tolist())
        inverse_labels_batch = separate_and_rescale_attributes_for_scoring(labels.tolist(), ids.tolist())

        if predictions.shape[-1] == 1:
            predictions = predictions.squeeze(-1)

        loss = self.mse_loss(predictions.float(), labels.float())
        
        return loss, inverse_predictions, inverse_labels_batch


    def evaluate(self, eval_loader, device="cuda"):
        self.eval()
        eval_total_loss = 0
        eval_inverse_label = []
        eval_inverse_pred = []

        with torch.no_grad():
            for document_single, chunked_documents, label, id_ ,lengths in eval_loader:
                document_single = document_single.to(device)

                eval_predictions = self.forward(document_single, chunked_documents, device, lengths)

                loss, inverse_predictions, inverse_labels = self.compute_loss(eval_predictions, label, id_)
                eval_total_loss += loss.item()
            
                eval_pred = eval_predictions.detach().cpu().numpy().reshape(-1)
                eval_inverse_pred.append(separate_and_rescale_attributes_for_scoring(eval_pred, id))
                eval_inverse_label.append(separate_and_rescale_attributes_for_scoring(label, id))
                # eval_inverse_pred.append(inverse_predictions)
                # eval_inverse_label.append(inverse_labels)
            eval_inverse_pred_flattened = [item for sublist in eval_inverse_pred for item in sublist]
            eval_inverse_label_flattened = [item for sublist in eval_inverse_label for item in sublist]

            test_eva_res = evaluation(eval_inverse_pred_flattened, eval_inverse_label_flattened)
            pearson_score = float(test_eva_res[7])
            qwk_score = float(test_eva_res[8])

            return eval_total_loss / len(eval_loader), qwk_score, pearson_score

    


