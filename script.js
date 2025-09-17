let model;
//cargar el modelo
async function loadModel() {
  const status = document.getElementById("result");
  status.innerText = "Cargando modelo...";
  try {
    model = await tf.loadLayersModel("./modelo_tfjs/model.json");
    status.innerText = "Modelo cargado ✅";
    document.getElementById("predict-btn").disabled = false;
  } catch (err) {
    status.innerText = "❌ Error cargando modelo (mira consola)";
    console.error("Error cargando el modelo:", err);
  }
}


loadModel();

// Subir imagen
document.getElementById("file-input").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (event) => {
      document.getElementById("preview").src = event.target.result;
    };
    reader.readAsDataURL(file);
  }
});
// Predecir
document.getElementById("predict-btn").addEventListener("click", async () => {
  const imgElement = document.getElementById("preview");

  if (!imgElement.src || !model) {
    alert("Primero sube o toma una foto y espera que el modelo cargue.");
    return;
  }

  let tensorImg = tf.browser.fromPixels(imgElement)
    .resizeNearestNeighbor([64, 64]) // 👈 cambia según tu modelo
    .toFloat()
    .div(255.0)
    .expandDims();

  const prediction = model.predict(tensorImg);
  const result = prediction.dataSync();

  document.getElementById("result").innerText = `Predicción: ${result}`;
});

