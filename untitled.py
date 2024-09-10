from transformers.utils import ModelOutput
from dataclasses import dataclass
import torch
from typing import Optional

@dataclass
class BertModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    scores: torch.FloatTensor = None
    label_ids: Optional[torch.LongTensor] = None  # 添加label_ids字段



from transformers import Trainer
class BERTTrainer(Trainer):

    def __init__(self, evaluator, train_sampler=None, *args, **kwargs):
        super(AESTrainer, self).__init__(*args, compute_metrics=self.compute_metrics, **kwargs)
        self.evaluator = evaluator
        self.dev_size = len(self.evaluator.dev_dataset)
        self.train_sampler = train_sampler
        
    def compute_metrics(self, p):
        preds, _ = p.predictions[1], p.label_ids
        dev_preds = preds[:self.dev_size]
        test_preds = preds[self.dev_size:]
        results = self.evaluator.evaluate(dev_preds, test_preds, self.state.epoch)
        return results