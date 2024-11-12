document.getElementById("form-comparar").onsubmit = async function(event) {
    event.preventDefault();

    const formData = new FormData();
    formData.append("imagen1", document.getElementById("imagen1").files[0]);
    formData.append("imagen2", document.getElementById("imagen2").files[0]);

    // Enviar las imágenes al servidor backend
    const response = await fetch("http://127.0.0.1:5000/comparar", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    // Mostrar resultados
    if (response.ok) {
        document.getElementById("resultados").style.display = "block";
        document.getElementById("similitud").textContent = data.similitud;
        document.getElementById("prediccion1").textContent = data.prediccion1;
        document.getElementById("prediccion2").textContent = data.prediccion2;
    } else {
        alert(data.error || "Ocurrió un error al procesar las imágenes.");
    }
};

async function obtenerGradCAM() {
    const formData = new FormData();
    formData.append("imagen", document.getElementById("imagen1").files[0]);  // Usamos imagen1 para el Grad-CAM

    const response = await fetch("http://127.0.0.1:5000/gradcam", {
        method: "POST",
        body: formData
    });

    if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        document.getElementById("gradcam-result").src = url;
        document.getElementById("gradcam-result").style.display = "block";
    } else {
        alert("Error al generar el Grad-CAM.");
    }
}
