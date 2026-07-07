(function () {
    const folderPathInput = document.getElementById('folderPath');
    const btnScan = document.getElementById('btnScan');
    const scanInfo = document.getElementById('scanInfo');
    const loading = document.getElementById('loading');
    const loadingText = document.getElementById('loadingText');
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');
    const totalFotos = document.getElementById('totalFotos');
    const totalPersonas = document.getElementById('totalPersonas');
    const totalSinRostro = document.getElementById('totalSinRostro');
    const folderList = document.getElementById('folderList');
    const outputPath = document.getElementById('outputPath');

    let scannedPath = '';
    let scannedCount = 0;

    function showLoading(text) {
        loadingText.textContent = text;
        loading.classList.remove('hidden');
        btnScan.disabled = true;
        folderPathInput.disabled = true;
        var orgBtn = document.querySelector('.btn-organize');
        if (orgBtn) orgBtn.disabled = true;
    }

    function hideLoading() {
        loading.classList.add('hidden');
        btnScan.disabled = false;
        folderPathInput.disabled = false;
        var orgBtn = document.querySelector('.btn-organize');
        if (orgBtn) orgBtn.disabled = !scannedPath;
    }

    function showError(message) {
        errorDiv.textContent = message;
        errorDiv.classList.remove('hidden');
        resultDiv.classList.add('hidden');
        hideLoading();
    }

    function showResult(data) {
        resultDiv.classList.remove('hidden');
        errorDiv.classList.add('hidden');

        totalFotos.textContent = data.total_fotos;
        totalPersonas.textContent = data.personas;
        totalSinRostro.textContent = data.sin_rostro;
        outputPath.textContent = data.output_path;

        folderList.innerHTML = '';
        data.carpetas.forEach(function (folder) {
            var item = document.createElement('li');
            item.textContent = folder + '/';
            folderList.appendChild(item);
        });

        if (data.advertencias && data.advertencias.length > 0) {
            var warnEl = document.createElement('div');
            warnEl.className = 'error';
            warnEl.style.marginTop = '12px';
            warnEl.textContent = 'Advertencias: ' + data.advertencias.join('; ');
            resultDiv.appendChild(warnEl);
        }
        hideLoading();
    }

    function doScan() {
        var path = folderPathInput.value.trim();
        if (!path) {
            showError('Escribe la ruta de la carpeta primero.');
            return;
        }
        showLoading('Escaneando carpeta...');
        errorDiv.classList.add('hidden');
        resultDiv.classList.add('hidden');
        scanInfo.classList.add('hidden');

        fetch('/api/scan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder_path: path })
        })
        .then(function (response) {
            return response.json().then(function (data) {
                if (!response.ok) throw new Error(data.error);
                return data;
            });
        })
        .then(function (data) {
            scannedPath = path;
            scannedCount = data.total;
            scanInfo.innerHTML =
                '<p><strong>' + data.total + '</strong> foto(s) válida(s) encontrada(s)</p>' +
                '<p class="scan-path">' + path + '</p>' +
                '<button id="btnOrganize" class="btn-organize">Organizar</button>';
            scanInfo.classList.remove('hidden');
            hideLoading();
            document.getElementById('btnOrganize').addEventListener('click', doOrganize);
        })
        .catch(function (err) {
            showError(err.message);
        });
    }

    function doOrganize() {
        if (!scannedPath) return;
        showLoading('Procesando imágenes (reconocimiento facial)...');
        errorDiv.classList.add('hidden');
        scanInfo.classList.add('hidden');

        fetch('/api/organize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder_path: scannedPath })
        })
        .then(function (response) {
            return response.json().then(function (data) {
                if (!response.ok) throw new Error(data.error);
                return data;
            });
        })
        .then(function (data) {
            showResult(data);
        })
        .catch(function (err) {
            showError(err.message);
        });
    }

    btnScan.addEventListener('click', doScan);

    folderPathInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') doScan();
    });
})();
