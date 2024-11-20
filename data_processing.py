import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Cargar y preparar los datos
def load_and_prepare_data(file_path, text_column, label_column):
    # Leer el archivo CSV
    df = pd.read_csv(file_path)
    
    # Obtener etiquetas únicas de la columna 'Type' y asignarlas a índices
    unique_labels = sorted(df[label_column].unique())  # Etiquetas únicas
    label_map = {label: idx for idx, label in enumerate(unique_labels)}  # Mapeo a índices
    
    # Convertir las etiquetas a índices numéricos
    df["labels"] = df[label_column].map(label_map)
    print(label_map)
    print(len(label_map))
    # Dividir los datos en entrenamiento y validación
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df[text_column].tolist(),
        df["labels"].tolist(),
        test_size=0.2,
        random_state=42,
    )
    
    return train_texts, val_texts, train_labels, val_labels, label_map

# Tokenizar y formatear los datos
def tokenize_and_format(texts, labels, tokenizer):
    dataset = Dataset.from_dict({"text": texts, "labels": labels})
    return dataset.map(
        lambda e: {
            **tokenizer(e["text"], truncation=True, padding='max_length', max_length=128),
            "labels": torch.tensor(e["labels"], dtype=torch.long),  # Las etiquetas son índices de clase
        },
        batched=True,
    )

