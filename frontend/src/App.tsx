import { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [backendStatus, setBackendStatus] = useState<string>("kontrol ediliyor...");

  // Sayfa açıldığında backend'in /health endpoint'ine ping at
  useEffect(() => {
    fetch("http://localhost:8000/health")
      .then((res) => res.json())
      .then((data) => setBackendStatus(`bağlı (${data.status})`))
      .catch(() => setBackendStatus("bağlantı yok"));
  }, []);

  return (
    <main className="container">
      <h1>Yerel RAG Asistanı</h1>
      <p>Backend: {backendStatus}</p>
    </main>
  );
}

export default App;