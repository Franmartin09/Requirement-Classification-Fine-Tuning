from sklearn.metrics import classification_report
import numpy as np
def evaluate_model(trainer, val_dataset, unique_labels):
    # Realiza la predicción
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)  # Obtener las predicciones con mayor probabilidad
    true_labels = predictions.label_ids

    # Verifica las clases predichas y las clases verdaderas
    print(f"Clases verdaderas: {set(true_labels)}")
    print(f"Clases predichas: {set(pred_labels)}")

    # Asegúrate de que el número de clases coincida
    if len(unique_labels) != len(set(pred_labels)):
        print("Número de clases no coincide entre las predicciones y las etiquetas únicas.")
        # Aquí puedes añadir clases que no fueron predichas
        all_labels = list(unique_labels.values())  # Asegúrate de que todas las clases están presentes

    # Reporte de clasificación (agregar explícitamente las clases)
    report = classification_report(true_labels, pred_labels, target_names=list(unique_labels.keys()), labels=all_labels)
    print(report)
