import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load trained model
model = load_model("models/garbage_classifier.h5")

# Define class labels (update based on your dataset folders)
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Start webcam
cap = cv2.VideoCapture(0)

print("Starting real-time garbage classification... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    img = cv2.resize(frame, (224, 224))
    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array)
    label_index = np.argmax(predictions[0])
    label = class_labels[label_index]
    confidence = predictions[0][label_index]

    # Display result
    display_text = f"{label.upper()} ({confidence*100:.2f}%)"
    cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("EcoVision Garbage Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
