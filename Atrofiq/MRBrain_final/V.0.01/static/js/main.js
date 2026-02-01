document.addEventListener('DOMContentLoaded', () => {
    // --- Elements ---
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const analyzeBtn = document.getElementById('btn-analyze');
    const logs = document.getElementById('log-terminal');
    const progressBar = document.getElementById('progress-bar');
    const progressArea = document.getElementById('progress-area');

    // Viewer Elements
    const viewerPlaceholder = document.getElementById('viewer-placeholder');
    const sliceImg = document.getElementById('slice-axial');

    // Metrics Elements
    const resPredAge = document.getElementById('res-pred-age');
    const resBag = document.getElementById('res-bag');

    // State
    let isUploading = false;
    let isAnalyzing = false;
    let pollInterval = null;

    // --- File Handling ---
    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('active'); // Add active style in CSS if needed
        dropzone.style.borderColor = 'var(--accent-primary)';
    });

    dropzone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropzone.classList.remove('active');
        dropzone.style.borderColor = 'var(--border-glass)';
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'var(--border-glass)';
        if (isUploading) return;
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
        if (isUploading) return;
        handleFiles(e.target.files);
    });

    function handleFiles(files) {
        if (files.length === 0) return;

        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('files[]', files[i]);
        }

        isUploading = true;
        // Visual feedback
        const originalContent = dropzone.innerHTML;
        dropzone.innerHTML = '<i class="fa fa-spinner fa-spin upload-icon"></i><p>Uploading...</p>';

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                isUploading = false;
                dropzone.innerHTML = `<i class="fa fa-check-circle upload-icon" style="color:var(--accent-success)"></i><p>Ready: ${data.count} files</p>`;

                // Show file list
                const fileList = document.getElementById('file-list');
                fileList.innerHTML = `<div class="file-item"><i class="fa-regular fa-folder"></i> Imported Session (${data.count} files)</div>`;

                analyzeBtn.disabled = false;

                // Auto Populate Metadata
                if (data.metadata) {
                    if (data.metadata.age) document.getElementById('input-age').value = data.metadata.age;
                    if (data.metadata.sex) document.getElementById('input-sex').value = data.metadata.sex;
                }
            })
            .catch(err => {
                isUploading = false;
                dropzone.innerHTML = originalContent;
                alert("Upload Failed");
                console.error(err);
            });
    }

    // --- Analysis ---
    analyzeBtn.addEventListener('click', () => {
        if (isAnalyzing) return;

        isAnalyzing = true;
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fa fa-circle-notch fa-spin"></i> Processing...';

        progressArea.classList.remove('hidden');
        logs.innerHTML = '> Starting pipeline...';

        // Reset Viewer
        sliceImg.src = '';
        sliceImg.classList.remove('active');
        viewerPlaceholder.style.display = 'flex';

        // Reset Stats
        resPredAge.innerText = '--';
        resBag.innerText = '--';
        resPredAge.classList.remove('highlight');

        const payload = {
            age: document.getElementById('input-age').value,
            sex: document.getElementById('input-sex').value,
            force_contrast: document.getElementById('force-contrast').checked
        };

        fetch('/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
            .then(res => res.json())
            .then(data => {
                pollInterval = setInterval(checkStatus, 1000);
            });
    });

    function checkStatus() {
        fetch('/status')
            .then(res => res.json())
            .then(data => {
                // Update Logs
                if (data.logs) {
                    logs.innerHTML = data.logs.map(l => `> ${l}`).join('<br>');
                    logs.scrollTop = logs.scrollHeight;
                }

                if (data.status === 'completed') {
                    clearInterval(pollInterval);
                    finishAnalysis();
                } else if (data.status === 'failed' || data.status === 'error') {
                    clearInterval(pollInterval);
                    alert("Analysis Failed. Check logs.");
                    resetUI();
                } else {
                    // Fake Progress
                    progressBar.style.width = '60%';
                }
            });
    }

    function resetUI() {
        isAnalyzing = false;
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fa-solid fa-bolt"></i> RUN ANALYSIS';
        progressBar.style.width = '0%';
        progressArea.classList.add('hidden');
    }

    function finishAnalysis() {
        progressBar.style.width = '100%';

        fetch('/results_data')
            .then(res => res.json())
            .then(data => {
                resetUI();
                analyzeBtn.innerHTML = '<i class="fa-solid fa-rotate-right"></i> RUN AGAIN';

                // 1. Stats
                if (data.predicted_age) {
                    resPredAge.innerText = data.predicted_age;
                    resPredAge.classList.add('highlight');
                    resBag.innerText = data.bag || '0.0';
                }

                // 2. Viewer
                // Hide placeholder, show image
                viewerPlaceholder.style.display = 'none';
                sliceImg.style.display = 'block';

                const slices = data.best_slices || { 'axial': 50 };
                // Add timestamp to prevent caching
                sliceImg.src = `/slice/axial/${slices.axial}?t=${new Date().getTime()}`;
                setTimeout(() => sliceImg.classList.add('active'), 50);

                // 3. Populate Regions Dropdown
                const regionSel = document.getElementById('region-select');
                regionSel.innerHTML = '<option>Select Region...</option>';

                const regions = data.available_regions || [];
                regions.forEach(r => {
                    const opt = document.createElement('option');
                    opt.value = r;
                    opt.textContent = r.replace(/_/g, ' ').toUpperCase();
                    regionSel.appendChild(opt);
                });

                // Auto-load if hippocampus exists
                if (regions.includes('left_hippocampus')) {
                    regionSel.value = 'left_hippocampus';
                    loadPlot('left_hippocampus');
                } else if (regions.length > 0) {
                    regionSel.value = regions[0];
                    loadPlot(regions[0]);
                }
            })
            .catch(e => {
                console.error(e);
                resetUI();
            });
    }

    // --- Charting ---
    document.getElementById('region-select').addEventListener('change', (e) => {
        if (e.target.value && e.target.value !== 'Select Region...') {
            loadPlot(e.target.value);
        }
    });

    async function loadPlot(region) {
        const timestamp = new Date().getTime();
        try {
            const resp = await fetch(`/normative_plot/${region}?t=${timestamp}`);
            if (resp.ok) {
                const data = await resp.json();
                renderChart(data, region);
            }
        } catch (e) { console.error(e); }
    }

    let normChart = null;
    function renderChart(data, region) {
        const ctx = document.getElementById('normative-chart').getContext('2d');
        if (normChart) normChart.destroy();

        Chart.defaults.font.family = "'Inter', sans-serif";
        Chart.defaults.color = "#94a3b8";
        Chart.defaults.borderColor = "rgba(148, 163, 184, 0.1)";

        normChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.ages,
                datasets: [
                    {
                        label: 'Median',
                        data: data.centiles['50'],
                        borderColor: '#10b981', // Emerald
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.4
                    },
                    {
                        label: 'Range (5-95%)',
                        data: data.centiles['95'],
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        fill: '+1',
                        pointRadius: 0,
                        borderWidth: 0,
                        tension: 0.4
                    },
                    {
                        data: data.centiles['5'], // Bounds (hidden line)
                        fill: false,
                        pointRadius: 0,
                        borderWidth: 0,
                        tension: 0.4
                    },
                    {
                        label: 'Patient',
                        data: new Array(data.ages.length).fill(null).map((_, i) =>
                            Math.abs(data.ages[i] - data.subject.age) < 1.0 ? data.subject.volume : null
                        ),
                        pointStyle: 'circle',
                        pointRadius: 6,
                        backgroundColor: '#ef4444',
                        borderColor: '#fff',
                        borderWidth: 2,
                        showLine: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.9)',
                        titleColor: '#f8fafc',
                        bodyColor: '#cbd5e1',
                        borderColor: 'rgba(148, 163, 184, 0.1)',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: { grid: { display: false } },
                    y: { grid: { color: 'rgba(148, 163, 184, 0.05)' } }
                }
            }
        });
    }

});
