#document_bert_architectures
import torch
from torch import nn
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertPreTrainedModel, BertConfig, BertModel
import torch.nn.functional as F
from data.scale import get_scaled_down_scores, separate_and_rescale_attributes_for_scoring
from utils.evaluate import evaluation
from utils.masks import get_trait_mask_by_prompt_ids
from tqdm import tqdm
from model.output import BertModelOutput


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


class SoftAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SoftAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.w = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v = nn.Linear(self.hidden_dim, 1)

    def forward(self, h):
        w = torch.tanh(self.w(h))
        weight = self.v(w)
        weight = weight.squeeze(-1)
        weight = torch.softmax(weight, 1)
        weight = weight.unsqueeze(-1)

        out = torch.mul(h, weight.repeat(1, 1, h.size(2)))
        out = torch.sum(out, 1)
        return out


class DocumentBertSentenceChunkAttentionLSTM(BertPreTrainedModel):
    def __init__(self, dim):
        bert_model_config = BertConfig.from_pretrained('bert-base-uncased')
        super(DocumentBertSentenceChunkAttentionLSTM, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        for layer in self.bert.encoder.layer[:11]:
            for param in layer.parameters():
                param.requires_grad = False
        # for param in self.bert.parameters():#212121
        #     param.requires_grad = False
        
        self.dropout = nn.Dropout(0.5)
        self.lstm = LSTM(bert_model_config.hidden_size, bert_model_config.hidden_size, batch_first = True)
        self.w_omega = nn.Parameter(torch.Tensor(bert_model_config.hidden_size, bert_model_config.hidden_size))
        self.b_omega = nn.Parameter(torch.Tensor(1, bert_model_config.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(bert_model_config.hidden_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        nn.init.uniform_(self.b_omega, -0.1, 0.1)
        self.linear = nn.Linear(bert_model_config.hidden_size, dim)

    def forward(self, document_batch: torch.Tensor, readability, hand_craft, device="cuda", bert_batch_size=0, length=0):
        bert_output = torch.zeros(size=(document_batch.shape[0],
                      document_batch.shape[1],
                      self.bert.config.hidden_size), dtype=torch.float, device=device)
        
        for doc_id in range(document_batch.shape[0]):
            bert_output_temp = self.bert(document_batch[doc_id][:length[doc_id], 0],
            token_type_ids = document_batch[doc_id][:length[doc_id], 1],
            attention_mask = document_batch[doc_id][:length[doc_id], 2])[1]

            bert_output_temp = torch.nan_to_num(bert_output_temp, nan=0.0)
            bert_output[doc_id][:length[doc_id]] = self.dropout(bert_output_temp)
        
        
        packed_input = pack_padded_sequence(bert_output, length, batch_first=True, enforce_sorted=False)
        packed_output, (_, _) = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)


        attention_w = torch.tanh(torch.matmul(output, self.w_omega) + self.b_omega)
        attention_u = torch.matmul(attention_w, self.u_omega)  # (batch_size, seq_len, 1)
        attention_score = F.softmax(attention_u, dim=1)  # (batch_size, seq_len, 1)
        attention_hidden = output * attention_score  # (batch_size, seq_len, num_hiddens)
        attention_hidden = torch.sum(attention_hidden, dim=1)  # (batch_size, num_hiddens)
        attention_hidden = self.linear(attention_hidden)
        
        return attention_hidden


class DocumentBertCombineWordDocumentLinear(BertPreTrainedModel):
    def __init__(self, dim):
        bert_model_config = BertConfig.from_pretrained('bert-base-uncased')
        super(DocumentBertCombineWordDocumentLinear, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        for layer in self.bert.encoder.layer[:11]:
            for param in layer.parameters():
                param.requires_grad = False
        # for param in self.bert.parameters():
        #     param.requires_grad = False
    
        self.bert_batch_size = 1
        self.dropout = nn.Dropout(0.5)
        self.mlp = nn.Sequential(
            nn.Linear(bert_model_config.hidden_size * 2, dim)
        )
        self.mlp.apply(init_weights)

    def forward(self, document_batch: torch.Tensor, readability, hand_craft, device="cuda"):

        # bert_output = torch.zeros(size=(document_batch.shape[0], 
        #               min(document_batch.shape[1], self.bert_batch_size), #分段的長度
        #               self.bert.config.hidden_size * 2),
        #               dtype=torch.float, device="cuda")
        # for doc_id in range(document_batch.shape[0]):
        #     # 只取分段一
        #     all_bert_output_info = self.bert(document_batch[doc_id][:self.bert_batch_size, 0],
        #                       token_type_ids=document_batch[doc_id][:self.bert_batch_size, 1],
        #                       attention_mask=document_batch[doc_id][:self.bert_batch_size, 2])
        #     bert_token_max = torch.max(all_bert_output_info[0], 1)
        #     bert_output[doc_id][:self.bert_batch_size] = torch.cat((bert_token_max.values, all_bert_output_info[1]), 1)
        # prediction = self.mlp(bert_output.view(bert_output.shape[0], -1))
        
        all_bert_output_info = self.bert(
                        document_batch[:, 0, 0],
                        token_type_ids = document_batch[:, 0, 1],
                        attention_mask = document_batch[:, 0, 2]
                )
        bert_token_max = torch.max(all_bert_output_info[0], 1)
        bert_output = torch.cat((all_bert_output_info[1], bert_token_max.values), 1)
        #mlp
        prediction = self.mlp(bert_output)
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction


class MultiTrait(nn.Module):
    def __init__(self, args):
        super(MultiTrait, self).__init__()
        dim = args.hidden_dim + 86 # 86 -> hand_craft + readability
        
        self.cross_atten = nn.MultiheadAttention(args.hidden_dim, args.mhd_head, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),  
            nn.ReLU(),
            nn.Dropout(p=0.3),  
            nn.Linear(dim, dim//2)  
        )

        
    def forward(self, doc_fea, chunk_fea, readability, hand_craft):
        cross_out, _ = self.cross_atten(doc_fea.unsqueeze(1), chunk_fea, chunk_fea)
        cross_fea = cross_out.squeeze(1)
        trait_fea = self.mlp(torch.cat([cross_fea, readability, hand_craft], dim=-1))

        return trait_fea


class Scorer(nn.Module):
    def __init__(self, args):
        super(Scorer, self).__init__()
        dim = (args.hidden_dim + 86) // 2 # 86 -> hand_craft + readability
        self.trait_att = nn.MultiheadAttention(dim, 1, batch_first=True)
        self.score_layer = nn.Linear(dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, mask):
        atten_out, _ = self.trait_att(x, x, x)
        atten_out = atten_out * mask.unsqueeze(-1)
        out = self.score_layer(atten_out).squeeze(-1)
        out = self.sigmoid(out)
        
        return out


class TraitSimilarity(nn.Module):

    def __init__(self, delta=0.7):
        super(TraitSimilarity, self).__init__()
        self.delta = delta
        
    def forward(self, y_pred, y_true):
        total_loss = torch.tensor(0.0, requires_grad=True, device=y_true.device)
        c = 0
        for j in range(1, y_true.size(1)):
            for k in range(j+1, y_true.size(1)):
                pcc = self.pearson_correlation_coefficient(y_true[:, j], y_true[:, k])
                cos = torch.cosine_similarity(y_pred[:, j], y_pred[:, k], dim=0)
                if pcc >= self.delta:
                    total_loss = total_loss + (1 - cos)
                c += 1
                
        return total_loss / c

    def pearson_correlation_coefficient(self, x, y):
        # Calculate the means of x and y
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        
        # Calculate the differences from the mean
        diff_x = x - mean_x
        diff_y = y - mean_y
        
        # Calculate the numerator and denominator for the correlation coefficient formula
        numerator = torch.sum(diff_x * diff_y)
        denominator_x = torch.sqrt(torch.sum(diff_x**2))
        denominator_y = torch.sqrt(torch.sum(diff_y**2))
        
        # Calculate the correlation coefficient
        r = numerator / (denominator_x * denominator_y)
        
        return r


class multiBert(nn.Module):

    def __init__(self, args):
        super(multiBert, self).__init__()
        self.chunk = DocumentBertSentenceChunkAttentionLSTM(args.hidden_dim)
        self.linear = DocumentBertCombineWordDocumentLinear(args.hidden_dim)
        self.multi_trait = nn.ModuleList([
            MultiTrait(args)
            for _ in range(args.num_trait)
        ])
        self.scorer = Scorer(args=args)
        self.hidden_dim = args.hidden_dim
        self.chunk_sizes = args.chunk_sizes
        self.mse_loss = nn.MSELoss()
        self.pooling = SoftAttention(args.hidden_dim)
        self.ts_loss = TraitSimilarity(args.delta)
        self.args = args
    
    def forward(self, prompt_id, document_single, chunked_documents, readability, hand_craft, scaled_score, device="cuda", lengths=0):
        prediction_single = self.linear(document_single, device=device, readability=readability, hand_craft=hand_craft) #(batch_size, 1)
        prediction_chunked = torch.empty(prediction_single.shape[0], 0, self.hidden_dim, device=device)
        
        for chunk_index in range(len(self.chunk_sizes)):
            batch_document_tensor_chunk = chunked_documents[chunk_index].to(device)
            length = lengths[chunk_index]
            length = length.cpu()
            predictions_chunk = self.chunk(batch_document_tensor_chunk, device=device, length=length, readability=readability, hand_craft=hand_craft)
            predictions_chunk = predictions_chunk.unsqueeze(1)
            prediction_chunked = torch.cat((prediction_chunked, predictions_chunk), dim=1)            
        
        trait_feas = torch.tensor([], requires_grad=True).to(device)
        for trait in self.multi_trait:
            trait_fea = trait(prediction_single, prediction_chunked, readability, hand_craft)
            trait_feas = torch.cat([trait_feas, trait_fea.unsqueeze(1)], dim=1)
            
        scaled_score = scaled_score.to(device)
        mask = get_trait_mask_by_prompt_ids(prompt_id).to(device)
        pred_scores = self.scorer(trait_feas, mask)
        mask = self._get_mask(scaled_score)
        pair_mask = ~mask
        
        mse_loss = self.mse_loss(pred_scores[mask], scaled_score[mask])
        ts_loss = self.ts_loss(
            pred_scores.masked_fill(pair_mask, -0.), 
            scaled_score.masked_fill(pair_mask, -0.)
        )
        beta = 0.2
        loss = 0.7 * mse_loss  + 0.3 * ts_loss
        
        return BertModelOutput(
            loss = loss,
            logits = pred_scores,
            scores = scaled_score
        )

    def _get_mask(self, target):
        mask = torch.ones(*target.size(), device=target.device)
        mask.data.masked_fill_((target == -1), 0)
        return mask.to(torch.bool)
