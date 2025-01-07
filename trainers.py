from transformers import Trainer


class BertTrainer(Trainer):

    def __init__(self, evaluator, train_sampler=None, test_dataset=None, *args, **kwargs):
        super(BertTrainer, self).__init__(*args, compute_metrics=self.compute_metrics, **kwargs)
        self.evaluator = evaluator
        self.train_sampler = train_sampler
        self.test_dataset = test_dataset
    
    def compute_metrics(self, p):
        preds, _ = p.predictions[0], p.label_ids
        dev_preds = preds
        test_preds = self.evaluate_test_dataset()
        results = self.evaluator.evaluate(dev_preds, test_preds, self.state.epoch)
        return results

    def evaluate_test_dataset(self):
        self._is_predicting = True  # 设置标志位，防止递归
        original_compute_metrics = self.compute_metrics  # 保存原始 compute_metrics
        self.compute_metrics = None  # 禁用 compute_metrics
        predictions = self.predict(self.test_dataset)  # 调用 predict
        self.compute_metrics = original_compute_metrics  # 恢复 compute_metrics
        self._is_predicting = False  # 恢复标志位
        test_preds = predictions.predictions[0]


        return test_preds