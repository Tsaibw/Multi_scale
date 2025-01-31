#document_bert_architectures
import torch
from torch import nn
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertPreTrainedModel, BertConfig, BertModel
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(7)


class DocumentBertSentenceChunkAttentionLSTM(BertPreTrainedModel):
    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertSentenceChunkAttentionLSTM, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
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

    def forward(self, document_batch: torch.Tensor, device=device, bert_batch_size=0, length=0):
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
    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertCombineWordDocumentLinear, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        self.bert_batch_size = 1
        self.dropout = nn.Dropout(0.1)

        self.mlp = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(bert_model_config.hidden_size * 2, 1),
            nn.Sigmoid()#
        )
        self.mlp.apply(init_weights)

    def forward(self, document_batch: torch.Tensor, device=device):
        bert_output = torch.zeros(size=(document_batch.shape[0], # batch_size
                      min(document_batch.shape[1], self.bert_batch_size), #分段的長度
                      self.bert.config.hidden_size * 2),
                      dtype=torch.float, device=device)
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