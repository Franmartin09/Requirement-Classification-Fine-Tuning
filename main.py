import os
os.TF_ENABLE_ONEDNN_OPTS = 0
from data_processing import load_and_prepare_data, tokenize_and_format
from model_training import load_model_and_tokenizer, configure_trainer
from evaluation import evaluate_model
from save_model import save_model
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments


def fine_tune_bert(file_path):
    # Variables del dataset
    text_column = "Requirement"
    label_column = "Type"
    
    # Cargar y preparar datos
    train_texts, val_texts, train_labels, val_labels, unique_labels = load_and_prepare_data(
        file_path, text_column, label_column
    )

    # Cargar el tokenizer y el modelo BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(unique_labels))


    # Tokenización y preparación de datasets
    train_dataset = tokenize_and_format(train_texts, train_labels, tokenizer)
    val_dataset = tokenize_and_format(val_texts, val_labels, tokenizer)

    # Configurar y entrenar el modelo
    trainer = configure_trainer(model, tokenizer, train_dataset, val_dataset)
    print("Fine-tuning el modelo...")
    trainer.train()

    # Evaluar el modelo
    print("\nEvaluación del modelo fine-tuned:")
    print(unique_labels)
    evaluate_model(trainer, val_dataset, unique_labels)

    # Retornar modelo, tokenizador, y etiquetas
    return model, tokenizer, trainer, unique_labels

if __name__ == "__main__":
    # Ruta al dataset
    file_path = "software_requirements_extended.csv"
    model, tokenizer, trainer, unique_labels = fine_tune_bert(file_path)
    save_model(trainer, tokenizer, model_name="fine_tuned_bert")