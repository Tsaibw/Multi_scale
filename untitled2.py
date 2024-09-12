#!/usr/bin/env python
# coding: utf-8
import fire
import os
import torch
import pickle
from transformers import TrainingArguments
from torch.utils.data import RandomSampler, DataLoader
from dataset import get_asap_dataset_with_topic as get_asap_dataset
from model.protact import ProTACTModel as Model
from model_config import ProTACTModelConfig as ModelConfig
from trainers import AESTrainer
from utils.callbacks import EvaluateRecord
from utils.general_utils import seed_all
from utils.multitask_evaluator_all_attributes import Evaluator
from safetensors.torch import load_file

def train(
    test_prompt_id: int = 1,
    experiment_tag: str = "test",
    seed: int = 11,
    num_train_epochs: int = 14,
    batch_size: int = 8,
    gradient_accumulation: int = 1,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    max_length: int = 512
):
    seed_all(seed)

    train_dataset, dev_dataset, test_dataset = get_asap_dataset(test_prompt_id)
    model_config = ModelConfig()
    model = Model(
        args=model_config, 
        pos_vocab=train_dataset.get_pos_vocab(),
        word_vocab=train_dataset.get_word_vocab()
    )
    # train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=len(train_dataset) // 2)
    eval_dataset = dev_dataset + test_dataset
    evaluator = Evaluator(dev_dataset, test_dataset, seed)

    output_dir = f"ckpts/Curriculum/Epoch_20/prompt_{test_prompt_id}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        logging_dir=f"logs/{experiment_tag}/Curriculum/Epoch_20/prompt_{test_prompt_id}",
        evaluation_strategy="epoch",
        label_names=["scaled_score"],
        save_strategy="epoch",
        save_total_limit=5,
        do_eval=True,
        load_best_model_at_end=True, 
        fp16=False,
        remove_unused_columns=True,
        metric_for_best_model="eval_test_avg",
        greater_is_better=True,
        seed=seed,
        data_seed=seed,
        ddp_find_unused_parameters=False
    )
            
    trainer = AESTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # train_sampler=train_sampler,
        eval_dataset=eval_dataset,
        evaluator=evaluator,
        callbacks=[EvaluateRecord(output_dir)],
    )

    print('Trainer is using device:', trainer.args.device)
    print(test_prompt_id)
    # trainer.train()
    trainer.train(resume_from_checkpoint = f"/home/tsaibw/ProTACT_pytorch/ckpts/Curriculum/prompt_{test_prompt_id}/checkpoint-epoch_10")



if __name__ == "__main__":
    # for i in range(2,9):
    #     train(test_prompt_id = i)
    
    model_config = ModelConfig()
    score = {}
    filename = 'data.pkl'
    for i in range(1,9):
        output, label = inference(i)
        score[f"P{i}:pred"] = output
        score[f"P{i}:score"] = label
    with open(filename, 'wb') as file:
        pickle.dump(score, file)