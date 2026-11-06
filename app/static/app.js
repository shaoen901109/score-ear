// 聽歌寫譜 - 前端邏輯（純原生 JS，不依賴任何套件或 CDN）
(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);

  const dropzone = $("#dropzone");
  const fileInput = $("#file-input");
  const pickFileBtn = $("#pick-file-btn");
  const selectedFileName = $("#selected-file-name");
  const startBtn = $("#start-btn");

  const optMode = $("#opt-mode");
  const optInstrumentHintWrap = $("#opt-instrument-hint-wrap");
  const optInstrumentHint = $("#opt-instrument-hint");
  const optBpm = $("#opt-bpm");
  const optTimeSignature = $("#opt-time-signature");
  const optQuantize = $("#opt-quantize");
  const optUseGpu = $("#opt-use-gpu");
  const optSuppressHarmonics = $("#opt-suppress-harmonics");
  const optSuppressHarmonicsWrap = $("#opt-suppress-harmonics-wrap");

  const progressCard = $("#progress-card");
  const progressBarFill = $("#progress-bar-fill");
  const progressMessage = $("#progress-message");

  const errorCard = $("#error-card");
  const errorMessage = $("#error-message");

  const resultCard = $("#result-card");
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabStaff = $("#tab-staff");
  const tabJianpu = $("#tab-jianpu");
  const tabInfo = $("#tab-info");
  const staffContainer = $("#staff-container");
  const jianpuContainer = $("#jianpu-container");
  const infoTableBody = $("#info-table-body");
  const warningsBox = $("#warnings-box");
  const warningsList = $("#warnings-list");

  const partSelectWrap = $("#part-select-wrap");
  const partSelect = $("#part-select");
  const zoomRange = $("#zoom-range");
  const zoomValue = $("#zoom-value");

  const dlMusicxml = $("#dl-musicxml");
  const dlMidi = $("#dl-midi");
  const dlJianpu = $("#dl-jianpu");
  const dlStaff = $("#dl-staff");
  const dlJson = $("#dl-json");

  let selectedFile = null;
  let currentJobId = null;
  let pollTimer = null;
  let currentScore = null;

  // ---- 檔案選擇 / 拖放 ------------------------------------------------

  pickFileBtn.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length > 0) {
      setSelectedFile(fileInput.files[0]);
    }
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      setSelectedFile(files[0]);
    }
  });

  function setSelectedFile(file) {
    selectedFile = file;
    selectedFileName.textContent = file.name;
    startBtn.disabled = false;
  }

  // ---- 模式切換：顯示/隱藏獨奏樂器提示 --------------------------------

  optMode.addEventListener("change", () => {
    optInstrumentHintWrap.hidden = optMode.value !== "solo";
    updateSuppressHarmonicsEnabled();
  });

  optInstrumentHint.addEventListener("change", updateSuppressHarmonicsEnabled);

  // 抑制泛音誤判只對「獨奏＋自動判斷樂器」或「多樂器」模式有效果（背後走 basic-pitch 複音轉譜）；
  // 其他情況（鋼琴／人聲／鼓等專用模型）沒有作用，UI 上停用但選項仍會照常送出。
  function updateSuppressHarmonicsEnabled() {
    const meaningful = optMode.value === "multi" || (optMode.value === "solo" && optInstrumentHint.value === "auto");
    optSuppressHarmonics.disabled = !meaningful;
    optSuppressHarmonicsWrap.classList.toggle("option-disabled", !meaningful);
  }

  updateSuppressHarmonicsEnabled();

  // ---- 分頁切換 ----------------------------------------------------------

  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const tab = btn.dataset.tab;
      tabStaff.hidden = tab !== "staff";
      tabJianpu.hidden = tab !== "jianpu";
      tabInfo.hidden = tab !== "info";
      partSelectWrap.hidden = !(tab === "jianpu" && currentScore && currentScore.parts && currentScore.parts.length > 1);
    });
  });

  // ---- 縮放 ----------------------------------------------------------

  zoomRange.addEventListener("input", () => {
    const scale = zoomRange.value;
    zoomValue.textContent = scale + "%";
    staffContainer.style.setProperty("--zoom", scale / 100);
    jianpuContainer.style.setProperty("--zoom", scale / 100);
    applyZoom(staffContainer, scale / 100);
    applyZoom(jianpuContainer, scale / 100);
  });

  function applyZoom(container, scale) {
    const svgs = container.querySelectorAll("svg");
    svgs.forEach((svg) => {
      svg.style.transform = `scale(${scale})`;
      svg.style.transformOrigin = "top center";
    });
  }

  // ---- 開始寫譜 ------------------------------------------------------

  startBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    hideError();
    resultCard.hidden = true;
    progressCard.hidden = false;
    setProgress(0, "上傳檔案中……");
    startBtn.disabled = true;

    const options = {
      mode: optMode.value,
      instrument_hint: optInstrumentHint.value,
      bpm_override: optBpm.value ? parseFloat(optBpm.value) : null,
      time_signature: optTimeSignature.value,
      quantize: optQuantize.value,
      use_gpu: optUseGpu.checked,
      suppress_harmonics: optSuppressHarmonics.checked,
    };

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("options", JSON.stringify(options));

    try {
      const resp = await fetch("/api/jobs", { method: "POST", body: formData });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || "上傳失敗");
      }
      currentJobId = data.job_id;
      startPolling();
    } catch (err) {
      showError(err.message || String(err));
      progressCard.hidden = true;
      startBtn.disabled = false;
    }
  });

  function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(pollStatus, 700);
    pollStatus();
  }

  async function pollStatus() {
    if (!currentJobId) return;
    try {
      const resp = await fetch(`/api/jobs/${currentJobId}`);
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || "查詢任務狀態失敗");
      }

      setProgress(data.progress || 0, data.message || "");

      if (data.status === "done") {
        clearInterval(pollTimer);
        pollTimer = null;
        await loadResult();
        progressCard.hidden = true;
        startBtn.disabled = false;
      } else if (data.status === "error") {
        clearInterval(pollTimer);
        pollTimer = null;
        showError(data.error || "處理失敗");
        progressCard.hidden = true;
        startBtn.disabled = false;
      }
    } catch (err) {
      clearInterval(pollTimer);
      pollTimer = null;
      showError(err.message || String(err));
      progressCard.hidden = true;
      startBtn.disabled = false;
    }
  }

  function setProgress(fraction, message) {
    const pct = Math.round(Math.max(0, Math.min(1, fraction)) * 100);
    progressBarFill.style.width = pct + "%";
    progressMessage.textContent = message;
  }

  function showError(msg) {
    errorMessage.textContent = msg;
    errorCard.hidden = false;
  }

  function hideError() {
    errorCard.hidden = true;
    errorMessage.textContent = "";
  }

  // ---- 結果載入與顯示 --------------------------------------------------

  async function loadResult(partIndex) {
    let url = `/api/jobs/${currentJobId}/result`;
    if (partIndex !== undefined && partIndex !== null) {
      url += `?part=${partIndex}`;
    }
    const resp = await fetch(url);
    const data = await resp.json();
    if (!resp.ok) {
      showError(data.error || "取得結果失敗");
      return;
    }

    currentScore = data.score;
    renderStaff(data.staff_svg || []);
    renderJianpu(data.jianpu_svg || "");
    renderInfo(data.score);
    setupPartSelect(data.score);
    setupDownloadLinks();

    resultCard.hidden = false;
  }

  function renderStaff(svgList) {
    staffContainer.innerHTML = "";
    svgList.forEach((svgText) => {
      const wrapper = document.createElement("div");
      wrapper.innerHTML = svgText;
      staffContainer.appendChild(wrapper);
    });
  }

  function renderJianpu(svgText) {
    jianpuContainer.innerHTML = svgText || "";
  }

  function renderInfo(score) {
    infoTableBody.innerHTML = "";
    if (!score) return;

    const rows = [
      ["BPM", score.bpm != null ? score.bpm : "-"],
      ["調性", score.key ? `${score.key.tonic} ${score.key.mode === "minor" ? "小調" : "大調"}` : "-"],
      ["拍號", score.time_signature ? `${score.time_signature.beats}/${score.time_signature.beat_type}` : "-"],
    ];

    (score.parts || []).forEach((part) => {
      let noteCount = 0;
      (part.measures || []).forEach((m) => {
        (m.notes || []).forEach((n) => {
          if (n.pitches && n.pitches.length > 0) noteCount += 1;
        });
      });
      rows.push([`聲部：${part.name}`, `${noteCount} 個音符`]);
    });

    rows.forEach(([label, value]) => {
      const tr = document.createElement("tr");
      const td1 = document.createElement("td");
      td1.textContent = label;
      const td2 = document.createElement("td");
      td2.textContent = value;
      tr.appendChild(td1);
      tr.appendChild(td2);
      infoTableBody.appendChild(tr);
    });

    const warnings = score.warnings || [];
    if (warnings.length > 0) {
      warningsList.innerHTML = "";
      warnings.forEach((w) => {
        const li = document.createElement("li");
        li.textContent = w;
        warningsList.appendChild(li);
      });
      warningsBox.hidden = false;
    } else {
      warningsBox.hidden = true;
    }
  }

  function setupPartSelect(score) {
    const parts = (score && score.parts) || [];
    if (parts.length <= 1) {
      partSelectWrap.hidden = true;
      return;
    }

    partSelect.innerHTML = "";
    const allOption = document.createElement("option");
    allOption.value = "";
    allOption.textContent = "全部";
    partSelect.appendChild(allOption);

    parts.forEach((part, idx) => {
      const opt = document.createElement("option");
      opt.value = String(idx);
      opt.textContent = part.name || part.id || `聲部 ${idx + 1}`;
      partSelect.appendChild(opt);
    });

    partSelect.onchange = async () => {
      const val = partSelect.value;
      await loadResult(val === "" ? null : parseInt(val, 10));
    };

    const activeTab = document.querySelector(".tab-btn.active");
    partSelectWrap.hidden = !(activeTab && activeTab.dataset.tab === "jianpu");
  }

  function setupDownloadLinks() {
    const base = `/api/jobs/${currentJobId}/download/`;
    dlMusicxml.href = base + "musicxml";
    dlMidi.href = base + "midi";
    dlJianpu.href = base + "jianpu.svg";
    dlStaff.href = base + "staff.svg";
    dlJson.href = base + "json";
  }
})();
