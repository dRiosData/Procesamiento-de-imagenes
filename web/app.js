// Referencias UI
const fileInput  = document.getElementById("file");
const btnDetect  = document.getElementById("btnDetect");
const confRange  = document.getElementById("conf");
const confVal    = document.getElementById("confVal");
const previewImg = document.getElementById("preview");
const vizImg     = document.getElementById("viz");
const jsonEl     = document.getElementById("json");
const msgEl      = document.getElementById("msg");

// Mostrar valor de confianza dinámico
confRange.addEventListener("input", () => {
  confVal.textContent = parseFloat(confRange.value).toFixed(2);
});

// Vista previa local
fileInput.addEventListener("change", () => {
  const f = fileInput.files?.[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = e => previewImg.src = e.target.result;
  reader.readAsDataURL(f);
});

// Helpers
const api = {
  detect: (file, conf) => {
    const form = new FormData();
    form.append("file", file);
    const url = `/api/v1/detect-image?conf_threshold=${conf}`;
    return fetch(url, { method: "POST", body: form })
      .then(r => r.json());
  },
  detectB64: (file, conf) => {
    const form = new FormData();
    form.append("file", file);
    const url = `/api/v1/detect-image-b64?conf_threshold=${conf}`;
    return fetch(url, { method: "POST", body: form })
      .then(r => r.json());
  },
  detectViz: (file, conf) => {
    const form = new FormData();
    form.append("file", file);
    const url = `/api/v1/detect-image-viz?conf_threshold=${conf}`;
    return fetch(url, { method: "POST", body: form })
      .then(r => r.blob());
  }
};

function setMsg(t) { msgEl.textContent = t ?? ""; }
function setJson(data) {
  jsonEl.textContent = JSON.stringify(data, null, 2);
}

btnDetect.addEventListener("click", async () => {
  setMsg("");
  setJson({});
  vizImg.src = "";

  const file = fileInput.files?.[0];
  if (!file) { setMsg("Seleccioná un archivo primero."); return; }
  const conf = parseFloat(confRange.value || "0.35");

  try {
    setMsg("Procesando…");

    // 1) JSON crudo (cajas, scores, etc.)
    const jsonDet = await api.detect(file, conf);
    setJson(jsonDet);

    // 2) Imagen con cajas (PNG)
    const blob = await api.detectViz(file, conf);
    const vizUrl = URL.createObjectURL(blob);
    vizImg.src = vizUrl;

    // 3) Respuesta con imagen base64 incluida (opcional)
    const b64Det = await api.detectB64(file, conf);
    // b64Det.image_b64 -> si quisieras, podés mostrarlo:
    // vizImg.src = "data:image/png;base64," + b64Det.image_b64;

    setMsg(`Listo. Detecciones: ${jsonDet?.count ?? 0}`);
  } catch (err) {
    console.error(err);
    setMsg("Error al detectar. Revisá la consola (F12) y el servidor.");
  }
});
