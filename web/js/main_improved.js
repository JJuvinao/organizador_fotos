(function () {
  let appState = {
    folderPath: null,
    results: null,
    currentPerson: null,
  };

  const folderInput = document.getElementById("folderInput");
  const folderPathText = document.getElementById("folderPathText");
  const folderPathBox = document.getElementById("folderPath");
  const processBtn = document.getElementById("processBtn");
  const qualitySlider = document.getElementById("qualitySlider");
  const qualityValue = document.getElementById("qualityValue");
  const progress = document.getElementById("progress");
  const progressFill = document.getElementById("progressFill");

  const resultsContainer = document.getElementById("resultsContainer");
  const totalPhotos = document.getElementById("totalPhotos");
  const totalPeople = document.getElementById("totalPeople");
  const withFace = document.getElementById("withFace");
  const noFace = document.getElementById("noFace");
  const personsGrid = document.getElementById("personsGrid");

  const finalizeBtn = document.getElementById("finalizeBtn");
  const startOverBtn = document.getElementById("startOverBtn");

  const successMessage = document.getElementById("successMessage");
  const outputPath = document.getElementById("outputPath");

  const personModal = document.getElementById("personModal");
  const modalPersonName = document.getElementById("modalPersonName");
  const modalPersonStats = document.getElementById("modalPersonStats");
  const modalPersonImages = document.getElementById("modalPersonImages");
  const modalCloseBtn = document.getElementById("modalCloseBtn");
  const modalClose = document.querySelector(".modal-close");

  folderInput.addEventListener("change", handleFolderSelect);
  qualitySlider.addEventListener("input", (e) => {
    qualityValue.textContent = parseFloat(e.target.value).toFixed(1);
  });
  processBtn.addEventListener("click", handleProcess);
  finalizeBtn.addEventListener("click", handleFinalize);
  startOverBtn.addEventListener("click", handleStartOver);
  modalClose.addEventListener("click", closeModal);
  modalCloseBtn.addEventListener("click", closeModal);
  personModal.addEventListener("click", (e) => {
    if (e.target === personModal) closeModal();
  });

  function handleFolderSelect(event) {
    const files = event.target.files;
    if (!files || files.length === 0) {
      showError("1", "No se seleccionó ninguna carpeta.");
      return;
    }
    const firstFile = files[0];
    const pathParts = firstFile.webkitRelativePath.split("/");
    const basePath = pathParts[0];
    appState.folderPath = basePath;
    folderPathText.textContent = basePath;
    folderPathBox.classList.remove("hidden");
    processBtn.disabled = false;
    clearError("1");
    goToStep(2);
  }

  async function handleProcess() {
    if (!appState.folderPath) {
      showError("2", "No se seleccionó carpeta.");
      return;
    }
    processBtn.disabled = true;
    showProgress(true);
    clearError("2");
    try {
      const response = await fetch("/api/organize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          folder_path: appState.folderPath,
          quality_threshold: parseFloat(qualitySlider.value),
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Error procesando carpeta");
      }
      appState.results = data;
      updateStats(data);
      renderPersons(data.personas_list);
      showProgress(false);
      goToStep(3);
    } catch (error) {
      showError("2", `Error: ${error.message}`);
      showProgress(false);
      processBtn.disabled = false;
    }
  }

  function updateStats(data) {
    totalPhotos.textContent = data.total_fotos;
    totalPeople.textContent = data.personas;
    withFace.textContent = data.fotos_con_rostro;
    noFace.textContent = data.sin_rostro;
    resultsContainer.classList.remove("hidden");
  }

  function renderPersons(personIds) {
    personsGrid.innerHTML = "";
    for (const personId of personIds) {
      if (personId === "sin_rostro") continue;
      const stats = appState.results.stats || {};
      const confidence = stats.cluster_confidence?.[personId] || 0;
      const imagesCount = appState.results.detalle?.[personId]?.count || 0;
      const card = document.createElement("div");
      card.className = "person-card";
      card.innerHTML = `
        <div class="person-name">${personId}</div>
        <div class="person-count">${imagesCount} foto(s)</div>
        <div class="person-confidence">
          Confianza: ${(confidence * 100).toFixed(0)}%
          <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${confidence * 100}%"></div>
          </div>
        </div>
      `;
      card.addEventListener("click", () => openPersonModal(personId));
      personsGrid.appendChild(card);
    }
  }

  async function openPersonModal(personId) {
    appState.currentPerson = personId;
    modalPersonName.textContent = personId;
    try {
      const response = await fetch(`/api/person_details/${personId}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      modalPersonStats.innerHTML = `<strong>📊 Estadísticas:</strong><p>• Total de fotos: ${data.image_count}</p><p>• Calidad: ${(data.embedding_quality * 100).toFixed(0)}%</p>`;
      modalPersonImages.innerHTML = "<strong>📁 Imágenes:</strong>" + data.images.map((img) => `<div class="modal-image-item">📷 ${img}</div>`).join("");
      personModal.classList.add("active");
    } catch (error) {
      alert(`Error: ${error.message}`);
    }
  }

  function closeModal() {
    personModal.classList.remove("active");
    appState.currentPerson = null;
  }

  async function handleFinalize() {
    finalizeBtn.disabled = true;
    try {
      const response = await fetch("/api/finalize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      outputPath.textContent = data.output_path;
      successMessage.classList.remove("hidden");
      goToStep(4);
    } catch (error) {
      alert(`Error: ${error.message}`);
      finalizeBtn.disabled = false;
    }
  }

  function handleStartOver() {
    appState = { folderPath: null, results: null, currentPerson: null };
    folderInput.value = "";
    folderPathBox.classList.add("hidden");
    processBtn.disabled = true;
    resultsContainer.classList.add("hidden");
    successMessage.classList.add("hidden");
    goToStep(1);
  }

  function goToStep(stepNum) {
    document.querySelectorAll(".step").forEach((s) => s.classList.remove("active"));
    document.getElementById(`step${stepNum}`).classList.add("active");
    setTimeout(() => {
      document.getElementById(`step${stepNum}`).scrollIntoView({ behavior: "smooth" });
    }, 100);
  }

  function showProgress(show) {
    progress.classList.toggle("hidden", !show);
    if (show) animateProgress();
  }

  function animateProgress() {
    let width = 0;
    const interval = setInterval(() => {
      width += Math.random() * 30;
      if (width > 90) width = 90;
      progressFill.style.width = width + "%";
      if (width >= 90) clearInterval(interval);
    }, 300);
  }

  function showError(step, message) {
    const errorDiv = document.getElementById(`error${step}`);
    errorDiv.textContent = message;
    errorDiv.classList.remove("hidden");
  }

  function clearError(step) {
    document.getElementById(`error${step}`).classList.add("hidden");
  }

  goToStep(1);
})();