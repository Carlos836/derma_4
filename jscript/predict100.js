// Solución al problema de exportación del módulo
import * as TargetClasses from './target_classes.js';

// Carga del modelo
let model;
(async function loadModel() {
    model = await tf.loadLayersModel('./model_kerasnative_v4/model.json');
    console.log('Modelo cargado correctamente');
})();

// Botón para cargar la imagen
const fileInput = document.getElementById('image-selector');
const uploadButton = document.getElementById('upload-button');
uploadButton.addEventListener('click', () => {
    fileInput.click();
});

// Selector de imagen y predicción
fileInput.addEventListener('change', async function () {
    const file = this.files[0];
    if (!file) return;

    // Muestra la imagen seleccionada
    const reader = new FileReader();
    reader.onload = function (e) {
        const imgElement = document.getElementById('uploaded-image');
        imgElement.src = e.target.result;
        imgElement.style.display = 'block';
    };
    reader.readAsDataURL(file);

    // Preprocesamiento de la imagen
    const imgTensor = await preprocessImage(file);

    // Realiza la predicción
    const prediction = model.predict(imgTensor);
    const predictedClass = prediction.argMax(-1).dataSync()[0];
    const confidence = prediction.max().dataSync()[0];

    // Muestra el resultado
    const resultText = `Prediction: ${TargetClasses[predictedClass]} (${(confidence * 100).toFixed(2)}%)`;
    document.getElementById('prediction').textContent = resultText;
    document.getElementById('prediction-container').style.display = 'block';
});

// Preprocesamiento de la imagen
async function preprocessImage(file) {
    const img = new Image();
    const reader = new FileReader();

    return new Promise((resolve) => {
        reader.onload = (e) => {
            img.src = e.target.result;
            img.onload = () => {
                // Resize y normalización
                const tensor = tf.browser.fromPixels(img)
                    .resizeNearestNeighbor([224, 224])
                    .toFloat()
                    .expandDims(0)
                    .div(255);
                resolve(tensor);
            };
        };
        reader.readAsDataURL(file);
    });
}

