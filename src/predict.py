import os
import sys
import argparse

import joblib
import numpy as np
import tensorflow as tf

from feature_extraction import extract_features_from_file


CLASS_NAMES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]


def predict_audio(file_path):
    # Find the project root
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    model_path = os.path.join(
        project_root, "models", "best_regularized_model.keras"
    )

    scaler_path = os.path.join(
        project_root, "models", "rich_feature_scaler.pkl"
    )

    # Load trained model and scaler
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    # Extract 86 features
    features = extract_features_from_file(file_path)

    if features.shape != (86,):
        raise ValueError("Expected exactly 86 audio features.")

    if not np.isfinite(features).all():
        raise ValueError("Audio features contain invalid values.")

    # Standardize using the training scaler
    features_scaled = scaler.transform(
        features.reshape(1, -1)
    ).astype(np.float32)

    # Predict sound category
    probabilities = model.predict(
        features_scaled, verbose=0
    )[0]

    predicted_class = int(np.argmax(probabilities))

    return CLASS_NAMES[predicted_class], float(
        probabilities[predicted_class]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify an urban sound recording"
    )

    parser.add_argument(
        "audio_file",
        help="Path to an audio recording"
    )

    args = parser.parse_args()

    label, confidence = predict_audio(args.audio_file)

    print("\nAUDIO CLASSIFICATION RESULT")
    print("-" * 40)
    print("Predicted sound:", label)
    print(f"Model probability: {confidence * 100:.2f}%")
