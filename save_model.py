import os

# Guardar el modelo y el tokenizador
def save_model(trainer, tokenizer, model_name="fine_tuned_bert"):
    output_dir = f"./{model_name}"

    # Crear el directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar el modelo y el tokenizador
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Modelo y tokenizador guardados en {output_dir}")
