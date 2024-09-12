from transformers import Trainer


class BertTrainer(Trainer):

    def __init__(self, evaluator, train_sampler=None, *args, **kwargs):
        super(BertTrainer, self).__init__(*args, compute_metrics=self.compute_metrics, **kwargs)
        self.evaluator = evaluator
        self.dev_size = len(self.evaluator.dev_dataset)
        self.train_sampler = train_sampler
        
    def compute_metrics(self, p):
        preds, _ = p.predictions[1], p.label_ids
        dev_preds = preds[:self.dev_size]
        test_preds = preds[self.dev_size:]
        results = self.evaluator.evaluate(dev_preds, test_preds, self.state.epoch)
        return results