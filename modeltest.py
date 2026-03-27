import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

model = tf.keras.models.load_model("asl_model.keras")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    "C:\\Users\\jvond\\ML_Project\\asl_alphabet_test",
    image_size=(200,200),   # replace with your actual size
    batch_size=32,
    shuffle=False
)

# evaluate overall accuracy
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.4f}")

# get true labels
y_true = np.concatenate([y.numpy() for x, y in test_dataset])

# get predictions
y_pred_probs = model.predict(test_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)

class_names = test_dataset.class_names

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=class_names))