(function () {
  const IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".svg",
    ".tiff",
    ".tif",
    ".avif",
  ];

  const folderInput = document.getElementById("folderInput");
  const resultDiv = document.getElementById("result");
  const countDisplay = document.getElementById("countDisplay");
  const countLabel = document.getElementById("countLabel");
  const errorDiv = document.getElementById("error");

  function isImageFile(file) {
    const ext = "." + file.name.split(".").pop().toLowerCase();
    return IMAGE_EXTENSIONS.includes(ext);
  }

  function pluralize(count) {
    return count === 1 ? "foto encontrada" : "fotos encontradas";
  }

  function showResult(count) {
    countDisplay.textContent = count;
    countLabel.textContent = pluralize(count);
    resultDiv.classList.remove("hidden");
    errorDiv.classList.add("hidden");
  }

  function showError(message) {
    errorDiv.textContent = message;
    errorDiv.classList.remove("hidden");
    resultDiv.classList.add("hidden");
  }

  function handleFolderSelect(event) {
    const files = event.target.files;

    if (!files || files.length === 0) {
      showError("No se seleccionó ninguna carpeta.");
      return;
    }

    const photoCount = Array.from(files).filter(isImageFile).length;

    if (photoCount === 0) {
      showError("No se encontraron fotos en la carpeta seleccionada.");
      return;
    }

    showResult(photoCount);
  }

  function handleError(event) {
    showError("Ocurrió un error al leer la carpeta.");
  }

  folderInput.addEventListener("change", handleFolderSelect);
  folderInput.addEventListener("error", handleError);
})();
