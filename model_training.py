from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import numpy as np
import torch

# Cargar el modelo y el tokenizador
def load_model_and_tokenizer(model_name, num_labels):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, problem_type="multi_label_classification"
    )
    return model, tokenizer

# Configuración del entrenador
def configure_trainer(model, tokenizer, train_dataset, val_dataset):
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=7,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    def compute_metrics(p):
        pred_labels = np.argmax(p.predictions, axis=1)
        accuracy = np.mean(pred_labels == p.label_ids)
        return {"accuracy": accuracy}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
   
    return trainer
