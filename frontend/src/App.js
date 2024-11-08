import React, { useState } from 'react';
import './App.css';

function App() {
  const [imagen, setImagen] = useState(null);
  const [prediccion, setPrediccion] = useState(null);
  const [cargando, setCargando] = useState(false);  // Nuevo estado de carga

  const manejarCambioImagen = (e) => {
    setImagen(e.target.files[0]);
  };

  const manejarEnvio = async (e) => {
    e.preventDefault();

    if (!imagen) {
      alert('Por favor, selecciona una imagen.');
      return;
    }

    setCargando(true);  // Mostrar indicador de carga
    const formData = new FormData();
    formData.append('file', imagen);

    try {
      const respuesta = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });

      const resultado = await respuesta.json();
      console.log(resultado);

      if (resultado.prediccion) {
        const indicePrediccionMasAlta = resultado.prediccion.indexOf(Math.max(...resultado.prediccion));
        const valorPrediccionMasAlta = resultado.prediccion[indicePrediccionMasAlta];
        setPrediccion({
          index: indicePrediccionMasAlta,
          value: valorPrediccionMasAlta,
        });

        alert(`Predicción: Clase ${indicePrediccionMasAlta} con una probabilidad de ${valorPrediccionMasAlta}`);
      } else {
        alert("Error en la predicción del modelo.");
      }
    } catch (error) {
      console.error('Error al conectar con el backend:', error);
    } finally {
      setCargando(false);  // Ocultar indicador de carga
      setImagen(null);     // Resetear imagen después del envío
    }
  };

  return (
    <div className="App">
      <h1>Cargar imagen</h1>
      <form onSubmit={manejarEnvio}>
        <input type="file" onChange={manejarCambioImagen} />
        <button type="submit">Confirmar</button>
      </form>

      {cargando && <p>Cargando...</p>}  {/* Mensaje de carga */}

      {prediccion && (
        <div>
          <h3>Predicción</h3>
          <p>Clase Predicha: {prediccion.index}</p>
          <p>Probabilidad: {prediccion.value}</p>
        </div>
      )}
    </div>
  );
}

export default App;
