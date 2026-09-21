/* ============================================================
   2mriky Cut Bulk — client-side batch background removal
   ------------------------------------------------------------
   THE QUALITY CONTRACT
   1. The uploaded file is decoded ONCE at full native resolution
      and is never resized, re-encoded or cropped.
   2. The AI model only ever sees a small downscaled COPY. It
      returns an alpha MASK, not an image.
   3. That mask is smoothly upscaled back to the original
      W x H and written into the alpha channel of the original
      pixels. Colour data is untouched.
   4. Export is PNG (lossless) or lossless WebP by default.
   => output dimensions === input dimensions, always.
   ============================================================ */

/* ---- CDN: try in order until one loads ---- */
const TRANSFORMERS_CDN = [
  'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3',
  'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.0',
  'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2',
];

/* ---- Per-model wiring. RMBG-1.4 is NOT a standard architecture:
        it needs an explicit model_type + processor config, or
        from_pretrained throws. ---- */
const MODELS = {
  'briaai/RMBG-1.4': {
    model: { config: { model_type: 'custom' } },
    processor: { config: {
      do_normalize: true, do_pad: false, do_rescale: true, do_resize: true,
      image_mean: [0.5, 0.5, 0.5], image_std: [1, 1, 1],
      feature_extractor_type: 'ImageFeatureExtractor',
      resample: 2, size: { width: 1024, height: 1024 },
    } },
    inputSize: 1024,
  },
  'Xenova/modnet': { model: {}, processor: {}, inputSize: 1024 },
};

let AutoModel, AutoProcessor, RawImage, env;

/* ============================================================
   1. STATE
   ============================================================ */
const state = {
  items: [],          // { id, file, name, w, h, srcUrl, outBlob, outUrl, status, error }
  busy: false,
  modelId: null,
  model: null,
  processor: null,
  device: 'wasm',
  seq: 0,
};

const MAX_SAFE_AREA = 16_777_216;   // ~16.7 MP — the iOS Safari canvas ceiling

/* ============================================================
   2. DOM
   ============================================================ */
const $ = (id) => document.getElementById(id);

/* Inline styles beat any class rule, so a cached stylesheet can never
   leave the modal stuck open again. */
function show(el, on) {
  if (!el) return;
  el.hidden = !on;
  el.style.display = on ? '' : 'none';
}
const dropzone   = $('dropzone');
const fileInput  = $('fileInput');
const gallery    = $('gallery');
const actionbar  = $('actionbar');
const runBtn     = $('runBtn');
const zipBtn     = $('zipBtn');
const clearBtn   = $('clearBtn');
const progWrap   = $('progressWrap');
const progFill   = $('progressFill');
const progLabel  = $('progressLabel');
const engineBadge= $('engineBadge');
const engineLabel= $('engineLabel');
const fmtSel     = $('fmt');
const bgModeSel  = $('bgMode');
const bgColorInp = $('bgColor');
const edgeInp    = $('edge');
const edgeOut    = $('edgeOut');
const modelSel   = $('model');

/* ============================================================
   3. BOOT — detect the best available compute device
   ============================================================ */
(async function boot() {
  let mod = null, lastErr = null;
  for (const url of TRANSFORMERS_CDN) {
    try { mod = await import(url); break; } catch (e) { lastErr = e; }
  }
  try {
    if (!mod) throw lastErr;
    ({ AutoModel, AutoProcessor, RawImage, env } = mod);
    env.allowLocalModels = false;
    env.useBrowserCache = true;
    try {
      if (!self.crossOriginIsolated) env.backends.onnx.wasm.numThreads = 1;
    } catch (_) { /* older builds */ }
  } catch (e) {
    setEngine('err', 'فشل تحميل المكتبة — راجع النت');
    console.error(e);
    return;
  }

  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) state.device = 'webgpu';
    } catch (_) { /* no webgpu */ }
  }

  if (state.device === 'webgpu') setEngine('gpu', 'WebGPU — سريع');
  else setEngine('ready', self.crossOriginIsolated ? 'WASM متعدد الأنوية' : 'WASM — أبطأ شوية');
})();

function setEngine(cls, text) {
  engineBadge.className = 'engine ' + cls;
  engineLabel.textContent = text;
}

/* ============================================================
   4. MODEL LOADING (lazy + cached by the browser)
   ============================================================ */
async function ensureModel(onProgress) {
  const id = modelSel.value;
  if (state.model && state.modelId === id) return;

  state.model = null;
  state.processor = null;
  state.modelId = null;

  const spec = MODELS[id] || { model: {}, processor: {} };

  // dtype fp32 is deliberate: the default is int8-quantized, which
  // visibly degrades the alpha matte. Quality is the whole point here.
  const load = async (device, dtype) => {
    const model = await AutoModel.from_pretrained(id, {
      ...spec.model, device, dtype, progress_callback: onProgress,
    });
    const processor = await AutoProcessor.from_pretrained(id, {
      ...spec.processor, progress_callback: onProgress,
    });
    return { model, processor };
  };

  const attempts = state.device === 'webgpu'
    ? [['webgpu', 'fp32'], ['wasm', 'fp32'], ['wasm', 'q8']]
    : [['wasm', 'fp32'], ['wasm', 'q8']];

  let loaded = null, lastErr = null;
  for (const [device, dtype] of attempts) {
    try {
      loaded = await load(device, dtype);
      state.device = device;
      if (dtype === 'q8') setEngine('ready', 'نسخة مضغوطة — جودة أقل');
      else if (device === 'wasm') setEngine('ready', 'WASM — دقة كاملة');
      break;
    } catch (e) {
      lastErr = e;
      console.warn(`load failed: ${device}/${dtype}`, e);
    }
  }
  if (!loaded) throw lastErr;

  state.model = loaded.model;
  state.processor = loaded.processor;
  state.modelId = id;
}

/* ============================================================
   5. INPUT HANDLING
   ============================================================ */
$('pickBtn').addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener('change', () => { addFiles(fileInput.files); fileInput.value = ''; });

['dragenter', 'dragover'].forEach(ev =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add('drag'); }));
['dragleave', 'drop'].forEach(ev =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove('drag'); }));
dropzone.addEventListener('drop', (e) => addFiles(e.dataTransfer.files));

document.addEventListener('paste', (e) => {
  if (e.clipboardData?.files?.length) addFiles(e.clipboardData.files);
});

async function addFiles(fileList) {
  const files = [...fileList].filter(f => f.type.startsWith('image/'));
  if (!files.length) return;

  for (const file of files) {
    const item = {
      id: ++state.seq,
      file,
      name: file.name || `pasted-${state.seq}.png`,
      w: 0, h: 0,
      srcUrl: URL.createObjectURL(file),
      outBlob: null, outUrl: null,
      status: 'queued', error: null,
    };
    state.items.push(item);
    renderCard(item);

    // read true dimensions without touching the pixels
    const img = new Image();
    img.onload = () => {
      item.w = img.naturalWidth;
      item.h = img.naturalHeight;
      updateCard(item);
      updateStats();
    };
    img.src = item.srcUrl;
  }
  updateStats();
  show(actionbar, true);
}

/* ============================================================
   6. THE CORE — process one image at full resolution
   ============================================================ */
async function processItem(item) {
  item.status = 'running';
  updateCard(item);

  // -- decode ONCE, at native resolution, no resampling --
  const bitmap = await createImageBitmap(item.file);
  const W = bitmap.width, H = bitmap.height;
  item.w = W; item.h = H;

  try {
    // -- build a small COPY purely as model input --
    const MODEL_IN = (MODELS[state.modelId]?.inputSize) || 1024;
    const scale = Math.min(1, MODEL_IN / Math.max(W, H));
    const sw = Math.max(1, Math.round(W * scale));
    const sh = Math.max(1, Math.round(H * scale));

    const small = document.createElement('canvas');
    small.width = sw; small.height = sh;
    const sctx = small.getContext('2d', { willReadFrequently: true });
    sctx.imageSmoothingEnabled = true;
    sctx.imageSmoothingQuality = 'high';
    sctx.drawImage(bitmap, 0, 0, sw, sh);
    const smallData = sctx.getImageData(0, 0, sw, sh);
    // .rgb() is required — the processor expects 3 channels, not RGBA.
    const rawSmall = new RawImage(new Uint8ClampedArray(smallData.data), sw, sh, 4).rgb();

    // -- inference: returns a MASK, never an image --
    const { pixel_values } = await state.processor(rawSmall);
    let out;
    try {
      out = await state.model({ input: pixel_values });
    } catch (_) {
      out = await state.model({ pixel_values });
    }
    const tensor = out.output ?? out.alpha ?? out.logits ?? Object.values(out)[0];

    // -- upscale the mask back to the ORIGINAL dimensions --
    const mask = await RawImage.fromTensor(tensor[0].mul(255).to('uint8')).resize(W, H);

    // -- write mask into the alpha channel of the untouched pixels --
    const canvas = document.createElement('canvas');
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);

    const imageData = ctx.getImageData(0, 0, W, H);
    const px = imageData.data;
    const m = mask.data;
    const lut = buildEdgeLUT(Number(edgeInp.value));
    const n = W * H;
    for (let i = 0; i < n; i++) px[(i << 2) + 3] = lut[m[i]];
    ctx.putImageData(imageData, 0, 0);

    // -- optional solid background, composited with correct alpha --
    let finalCanvas = canvas;
    const bg = resolveBg();
    if (bg) {
      const flat = document.createElement('canvas');
      flat.width = W; flat.height = H;
      const fctx = flat.getContext('2d');
      fctx.fillStyle = bg;
      fctx.fillRect(0, 0, W, H);
      fctx.drawImage(canvas, 0, 0);
      finalCanvas = flat;
    }

    // -- export --
    item.outBlob = await exportCanvas(finalCanvas);
    if (item.outUrl) URL.revokeObjectURL(item.outUrl);
    item.outUrl = URL.createObjectURL(item.outBlob);
    item.status = 'done';
    item.error = null;
  } catch (err) {
    console.error(err);
    item.status = 'error';
    item.error = err?.message || 'فشل غير معروف';
  } finally {
    bitmap.close?.();
    updateCard(item);
    updateStats();
  }
}

/* Alpha response curve: 0 = raw soft matte, 100 = hard cut. */
function buildEdgeLUT(strength) {
  const lut = new Uint8Array(256);
  const k = 1 + (strength / 100) * 16;
  const f = (x) => 1 / (1 + Math.exp(-k * (x - 0.5) * 2));
  const lo = f(0), hi = f(1);
  for (let i = 0; i < 256; i++) {
    const v = (f(i / 255) - lo) / (hi - lo);
    lut[i] = Math.max(0, Math.min(255, Math.round(v * 255)));
  }
  return lut;
}

function resolveBg() {
  const v = bgModeSel.value;
  if (v === 'transparent') return null;
  if (v === 'custom') return bgColorInp.value;
  return v;
}

function exportCanvas(canvas) {
  const v = fmtSel.value;
  return new Promise((resolve, reject) => {
    const done = (b) => b ? resolve(b) : reject(new Error('فشل التصدير'));
    if (v === 'image/png')            canvas.toBlob(done, 'image/png');
    else if (v === 'image/webp-lossless') canvas.toBlob(done, 'image/webp', 1);
    else                              canvas.toBlob(done, 'image/webp', 0.92);
  });
}

function outExt() {
  return fmtSel.value === 'image/png' ? 'png' : 'webp';
}
// الاسم الأصلي بالظبط — الامتداد بس هو اللي بيتغير حسب صيغة الحفظ.
function outName(item) {
  const base = item.name.replace(/\.[^.]+$/, '');
  return `${base}.${outExt()}`;
}

/* ============================================================
   7. BATCH RUNNER
   ============================================================ */
runBtn.addEventListener('click', async () => {
  if (state.busy) return;
  const queue = state.items.filter(i => i.status !== 'done');
  if (!queue.length) return;

  state.busy = true;
  runBtn.disabled = true;
  clearBtn.disabled = true;
  show(progWrap, true);
  setProgress(0, 'بيحمّل الموديل…');

  try {
    await ensureModel((p) => {
      if (p.status === 'progress' && p.total) {
        setProgress((p.loaded / p.total) * 100, `تحميل الموديل — ${(p.loaded / 1048576).toFixed(1)} / ${(p.total / 1048576).toFixed(1)} MB`);
      }
    });
  } catch (e) {
    console.error(e);
    setProgress(0, 'فشل تحميل الموديل — راجع الاتصال');
    setEngine('err', 'الموديل مش متحمّل');
    state.busy = false; runBtn.disabled = false; clearBtn.disabled = false;
    return;
  }

  for (let i = 0; i < queue.length; i++) {
    setProgress((i / queue.length) * 100, `بيعالج ${i + 1} من ${queue.length} — ${queue[i].name}`);
    await processItem(queue[i]);
    await new Promise(r => setTimeout(r, 0));   // let the UI breathe
  }

  setProgress(100, `خلص — ${state.items.filter(i => i.status === 'done').length} صورة جاهزة`);
  state.busy = false;
  runBtn.disabled = false;
  clearBtn.disabled = false;
  updateStats();
});

function setProgress(pct, label) {
  progFill.style.width = Math.max(0, Math.min(100, pct)) + '%';
  progLabel.textContent = label;
}

/* ============================================================
   8. DOWNLOADS
   ============================================================ */
function downloadBlob(blob, filename) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}

zipBtn.addEventListener('click', async () => {
  const done = state.items.filter(i => i.status === 'done' && i.outBlob);
  if (!done.length) return;

  zipBtn.disabled = true;
  const prev = zipBtn.textContent;
  zipBtn.textContent = 'بيحزم…';

  try {
    const zip = new JSZip();
    const used = new Map();
    for (const item of done) {
      let name = outName(item);
      const c = (used.get(name) || 0) + 1;
      used.set(name, c);
      if (c > 1) name = name.replace(/(\.[^.]+)$/, `-${c}$1`);
      // STORE: PNG/WebP are already compressed — deflate only wastes time.
      zip.file(name, item.outBlob, { compression: 'STORE' });
    }
    const blob = await zip.generateAsync({ type: 'blob' });
    const stamp = new Date().toISOString().slice(0, 10);
    downloadBlob(blob, `2mriky-cut-bulk-${stamp}.zip`);
  } catch (e) {
    console.error(e);
    alert('فشل تجهيز الـ ZIP — جرّب تنزّلهم واحدة واحدة.');
  } finally {
    zipBtn.textContent = prev;
    zipBtn.disabled = false;
  }
});

clearBtn.addEventListener('click', () => {
  if (state.busy) return;
  state.items.forEach(i => {
    URL.revokeObjectURL(i.srcUrl);
    if (i.outUrl) URL.revokeObjectURL(i.outUrl);
  });
  state.items = [];
  gallery.innerHTML = '';
  show(actionbar, false);
  show(progWrap, false);
  updateStats();
});

/* ============================================================
   9. RENDERING
   ============================================================ */
function renderCard(item) {
  const card = document.createElement('article');
  card.className = 'card';
  card.id = `card-${item.id}`;
  card.innerHTML = `
    <div class="thumb" data-role="thumb">
      <span class="badge" data-role="badge">في الطابور</span>
      <button class="card-remove" data-role="remove" aria-label="شيل">&times;</button>
      <img data-role="img" alt="">
    </div>
    <div class="card-body">
      <p class="card-name" data-role="name"></p>
      <p class="card-dims" data-role="dims"></p>
    </div>
    <div class="card-foot">
      <button data-role="dl" disabled>نزّل</button>
      <button data-role="cmp" disabled>قارن</button>
    </div>`;
  gallery.appendChild(card);

  card.querySelector('[data-role=remove]').addEventListener('click', () => {
    if (state.busy) return;
    URL.revokeObjectURL(item.srcUrl);
    if (item.outUrl) URL.revokeObjectURL(item.outUrl);
    state.items = state.items.filter(i => i.id !== item.id);
    card.remove();
    updateStats();
    if (!state.items.length) show(actionbar, false);
  });
  card.querySelector('[data-role=dl]').addEventListener('click', () => {
    if (item.outBlob) downloadBlob(item.outBlob, outName(item));
  });
  card.querySelector('[data-role=cmp]').addEventListener('click', () => openCompare(item));
  card.querySelector('[data-role=thumb]').addEventListener('click', (e) => {
    if (e.target.dataset.role === 'remove') return;
    if (item.status === 'done') openCompare(item);
  });

  updateCard(item);
}

function updateCard(item) {
  const card = $(`card-${item.id}`);
  if (!card) return;
  const badge = card.querySelector('[data-role=badge]');
  const img   = card.querySelector('[data-role=img]');
  const thumb = card.querySelector('[data-role=thumb]');
  const dims  = card.querySelector('[data-role=dims]');

  card.querySelector('[data-role=name]').textContent = item.name;

  const labels = { queued: 'في الطابور', running: 'شغّال…', done: 'تمام', error: 'فشل' };
  const cls    = { queued: '', running: 'run', done: 'done', error: 'err' };
  badge.textContent = labels[item.status];
  badge.className = 'badge ' + cls[item.status];
  card.classList.toggle('err', item.status === 'error');

  img.src = item.status === 'done' && item.outUrl ? item.outUrl : item.srcUrl;
  img.style.opacity = item.status === 'running' ? '.25' : '1';

  let spinner = thumb.querySelector('.spinner');
  if (item.status === 'running' && !spinner) {
    spinner = document.createElement('div');
    spinner.className = 'spinner';
    spinner.style.cssText = 'position:absolute;z-index:1';
    thumb.appendChild(spinner);
  } else if (item.status !== 'running' && spinner) {
    spinner.remove();
  }

  if (item.status === 'error') {
    dims.innerHTML = `<span style="color:var(--danger)">${escapeHtml(item.error || '')}</span>`;
  } else if (item.w) {
    const size = item.outBlob ? ` · ${fmtBytes(item.outBlob.size)}` : '';
    const keep = item.status === 'done' ? ` <span class="keep">✓ نفس المقاس</span>` : '';
    const warn = item.w * item.h > MAX_SAFE_AREA
      ? ` <span style="color:var(--warn)" title="ممكن تفشل على iOS">⚠ ${(item.w * item.h / 1e6).toFixed(0)}MP</span>` : '';
    dims.innerHTML = `${item.w}×${item.h}${size}${keep}${warn}`;
  } else {
    dims.textContent = '…';
  }

  card.querySelector('[data-role=dl]').disabled  = item.status !== 'done';
  card.querySelector('[data-role=cmp]').disabled = item.status !== 'done';
}

function updateStats() {
  const total = state.items.length;
  const done  = state.items.filter(i => i.status === 'done').length;
  const bytes = state.items.reduce((s, i) => s + (i.outBlob?.size || 0), 0);
  $('statCount').textContent = `${total} صورة`;
  $('statDone').textContent  = `${done} خلصت`;
  $('statSize').textContent  = bytes ? fmtBytes(bytes) : '—';
  zipBtn.disabled = done === 0 || state.busy;
  runBtn.disabled = state.busy || total === 0;
}

function fmtBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(0) + ' KB';
  return (b / 1048576).toFixed(1) + ' MB';
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* ============================================================
   10. COMPARE MODAL
   ============================================================ */
const modal = $('modal');
const cmpAfter = $('cmpAfter');
const cmpBefore = $('cmpBefore');
const cmpBeforeWrap = $('cmpBeforeWrap');
const cmpHandle = $('cmpHandle');
const compare = $('compare');

function openCompare(item) {
  if (item.status !== 'done') return;
  cmpAfter.src = item.outUrl;
  cmpBefore.src = item.srcUrl;
  $('modalMeta').textContent =
    `${item.name}  ·  ${item.w}×${item.h}  →  ${item.w}×${item.h}  ·  ${fmtBytes(item.outBlob.size)}`;
  show(modal, true);
  cmpAfter.onload = () => { syncCompareWidth(); setSplit(50); };
  if (cmpAfter.complete) { syncCompareWidth(); setSplit(50); }
}
function syncCompareWidth() {
  compare.style.setProperty('--cw', compare.clientWidth + 'px');
}
function setSplit(pct) {
  pct = Math.max(0, Math.min(100, pct));
  cmpBeforeWrap.style.width = pct + '%';
  cmpHandle.style.insetInlineStart = pct + '%';
}
function closeModal() { show(modal, false); cmpAfter.src = ''; cmpBefore.src = ''; }

$('modalClose').addEventListener('click', closeModal);
modal.addEventListener('click', (e) => { if (e.target === modal) closeModal(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !modal.hidden) closeModal(); });
window.addEventListener('resize', () => { if (!modal.hidden) syncCompareWidth(); });

let dragging = false;
const onMove = (e) => {
  if (!dragging) return;
  const r = compare.getBoundingClientRect();
  const x = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
  setSplit((x / r.width) * 100);
};
compare.addEventListener('pointerdown', (e) => { dragging = true; onMove(e); });
window.addEventListener('pointermove', onMove);
window.addEventListener('pointerup', () => { dragging = false; });

/* ============================================================
   11. SETTINGS UI
   ============================================================ */
const edgeLabel = (v) => v < 20 ? 'ناعمة جدًا' : v < 45 ? 'متوسطة' : v < 75 ? 'حادة' : 'قاطعة';
edgeInp.addEventListener('input', () => { edgeOut.textContent = edgeLabel(Number(edgeInp.value)); });

bgModeSel.addEventListener('change', () => {
  show(bgColorInp, bgModeSel.value === 'custom');
});

fmtSel.addEventListener('change', () => {
  const hints = {
    'image/png': 'PNG بيحفظ كل بكسل زي ما هو + الشفافية.',
    'image/webp-lossless': 'WebP بدون فقد — نفس البكسلات بحجم أقل بكتير.',
    'image/webp': 'WebP بجودة 92% — أخف حاجة، فقد بسيط جدًا مش بيتشاف.',
  };
  $('fmtHint').textContent = hints[fmtSel.value];
});

modelSel.addEventListener('change', () => {
  state.model = null;
  state.processor = null;
  state.modelId = null;
});

$('resetSettings').addEventListener('click', () => {
  fmtSel.value = 'image/png';
  bgModeSel.value = 'transparent';
  show(bgColorInp, false);
  edgeInp.value = 35;
  edgeOut.textContent = edgeLabel(35);
  fmtSel.dispatchEvent(new Event('change'));
});


/* حالة ابتدائية مضمونة مهما كان الـ CSS متكاش */
show(modal, false);
show(actionbar, false);
show(progWrap, false);
show(bgColorInp, false);
updateStats();
