let model;
const CLASS_NAMES = ["bengal_cat", "maine_coon_cat", "persian_cat", "siamese_cat"];

// Esperar a que TensorFlow.js se cargue completamente
function waitForTFJS() {
    return new Promise((resolve) => {
        if (typeof tf !== 'undefined') {
            resolve();
        } else {
            setTimeout(() => waitForTFJS().then(resolve), 100);
        }
    });
}

async function loadModel() {
    const status = document.getElementById("result");
    status.innerText = "Cargando modelo...";
    
    try {
        console.log("🎯 Intentando cargar modelo...");
        console.log("📦 Versión de TensorFlow.js:", tf.version.tfjs);
        
        model = await tf.loadLayersModel("modelo_tfjs/model.json");
        status.innerText = "✅ Modelo cargado correctamente";
        console.log("✅ Modelo cargado exitosamente");
        document.getElementById("predict-btn").disabled = false;
        
    } catch (err) {
        console.error("❌ Error detallado al cargar modelo:", err);
        status.innerText = "❌ Error cargando modelo. Ver consola (F12)";
    }
}

function preprocessImage(imgElement) {
    return tf.tidy(() => {
        let tensor = tf.browser.fromPixels(imgElement)
            .resizeNearestNeighbor([128, 128])
            .toFloat()
            .div(tf.scalar(255.0))
            .expandDims(0);
        
        console.log("📐 Tensor shape:", tensor.shape);
        return tensor;
    });
}

// Configurar event listeners después de que todo esté cargado
function setupEventListeners() {
    document.getElementById("predict-btn").addEventListener("click", async () => {
        const imgElement = document.getElementById("preview");
        
        if (!imgElement.src || imgElement.src === window.location.href) {
            alert("⚠️ Primero sube una imagen de un gato");
            return;
        }
        
        if (!model) {
            alert("⏳ El modelo aún está cargando...");
            return;
        }

        try {
            document.getElementById("result").innerText = "🔍 Analizando imagen...";
            
            const tensor = preprocessImage(imgElement);
            const prediction = model.predict(tensor);
            const probabilities = prediction.dataSync();
            
            console.log("📊 Probabilidades:", probabilities);
            
            let maxIndex = 0;
            let maxConfidence = 0;
            
            for (let i = 0; i < probabilities.length; i++) {
                if (probabilities[i] > maxConfidence) {
                    maxConfidence = probabilities[i];
                    maxIndex = i;
                }
            }
            
            const resultText = `🎯 Predicción: ${CLASS_NAMES[maxIndex]} (${(maxConfidence * 100).toFixed(1)}% de confianza)`;
            document.getElementById("result").innerText = resultText;
            
            // Debug info
            console.log("--- Todas las probabilidades ---");
            CLASS_NAMES.forEach((className, index) => {
                console.log(`${className}: ${(probabilities[index] * 100).toFixed(2)}%`);
            });
            
            tensor.dispose();
            prediction.dispose();
            
        } catch (error) {
            console.error("Error en predicción:", error);
            document.getElementById("result").innerText = "❌ Error en la predicción";
        }
    });

    document.getElementById("file-input").addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                const img = document.getElementById("preview");
                img.src = event.target.result;
                document.getElementById("result").innerText = "📷 Imagen lista para predecir";
            };
            reader.readAsDataURL(file);
        }
    });
}

// Iniciar aplicación cuando todo esté listo
async function initApp() {
    console.log("🚀 Iniciando aplicación...");
    
    // Esperar a que TensorFlow.js se cargue
    await waitForTFJS();
    console.log("✅ TensorFlow.js cargado");
    console.log("📦 Versión:", tf.version.tfjs);
    
    // Configurar event listeners
    setupEventListeners();
    
    // Cargar modelo
    await loadModel();
}

// Iniciar cuando el DOM esté listo
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}