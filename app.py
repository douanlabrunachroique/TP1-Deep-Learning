from flask import Flask, request, jsonify
import numpy as np
from tensorflow import keras

# Initialisation de l'application Flask
app = Flask(__name__)

# Chargement du modèle Keras (assure-toi que le fichier mnist_model.h5 est dans le même dossier)
MODEL_PATH = "mnist_model.h5"
model = keras.models.load_model(MODEL_PATH)
print(" Modèle chargé avec succès !")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Vérification que l'image est présente dans la requête
    if data is None or 'image' not in data:
        return jsonify({'error': 'Aucune image fournie. Format attendu : { "image": [784 valeurs] }'}), 400

    try:
        # Conversion en tableau numpy
        image_data = np.array(data['image'])
        # Reshape en 1x784
        image_data = image_data.reshape(1, 784)
        # Normalisation
        image_data = image_data.astype("float32") / 255.0

        # Prédiction avec le modèle
        prediction = model.predict(image_data)
        predicted_class = int(np.argmax(prediction, axis=1)[0])

        return jsonify({
            'prediction': predicted_class,
            'probabilities': prediction.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
