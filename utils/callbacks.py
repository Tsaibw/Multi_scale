import os

from transformers import TrainerCallback


class EvaluateRecord(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        self.best_dev_avg = -1
        self.best_dev_test_avg = -1
        self.best_test_avg = -1
        self.current_epoch = 1
        
    def on_save(self, args, state, control, **kwargs):
        last_result = state.log_history[-1]
        dev_avg = last_result["eval_dev_avg"]
        test_avg = last_result["eval_test_avg"]

        if self.best_dev_avg < dev_avg:
            self.best_dev_avg = dev_avg
            self.best_dev_test_avg = test_avg
        
        self.best_test_avg = max(self.best_test_avg, test_avg)

        with open(os.path.join(self.log_path, "final_report.txt"), "a") as w:
            w.write(f"-------------------- Epoch-{self.current_epoch} --------------------\n")
            w.write(f"[BEST DEV]\tAVG QWK: {self.best_dev_avg}\n")
            w.write(f"[BEST DEV-TEST]\tAVG QWK: {self.best_dev_test_avg}\n")
            w.write(f"[BEST TEST]\tAVG QWK: {self.best_test_avg}\n")
            self.current_epoch += 1