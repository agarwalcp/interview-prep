/* Browser-side sensing for InterviewIQ — plain signal processing, no ML models, nothing leaves your computer.
   Stillness  : frame-to-frame pixel change of the camera image (centre crop).
   Steadiness : how much the voice pitch (autocorrelation estimate) wobbles while you speak.            */

function rms(buf) {
  let s = 0; for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
  return Math.sqrt(s / buf.length);
}

/* normalised autocorrelation pitch estimate; returns {freq, clarity} or null */
function estimatePitch(buf, sr, fmin = 70, fmax = 400) {
  const n = buf.length;
  let e0 = 0; for (let i = 0; i < n; i++) e0 += buf[i] * buf[i];
  if (e0 < 1e-6) return null;
  const minLag = Math.floor(sr / fmax), maxLag = Math.min(Math.floor(sr / fmin), n - 2);
  const corr = new Float32Array(maxLag + 2);
  let best = 0;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let s = 0; for (let i = 0; i < n - lag; i++) s += buf[i] * buf[i + lag];
    corr[lag] = s / e0;
    if (corr[lag] > best) best = corr[lag];
  }
  if (best < 0.4) return null;                              // not clearly voiced
  for (let lag = minLag + 1; lag < maxLag; lag++) {         // smallest-lag peak close to the max (avoids octave errors)
    if (corr[lag] >= 0.9 * best && corr[lag] >= corr[lag - 1] && corr[lag] >= corr[lag + 1]) {
      return { freq: sr / lag, clarity: corr[lag] };
    }
  }
  return null;
}

/* mean absolute grey-level change between two equal-length frames, 0 (identical) .. 1 */
function frameMotion(prev, cur) {
  let d = 0; for (let i = 0; i < cur.length; i++) d += Math.abs(cur[i] - prev[i]);
  return d / cur.length / 255;
}

function mean(a) { return a.reduce((x, y) => x + y, 0) / a.length; }
function median(a) { const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }
function stdev(a) { const m = mean(a); return Math.sqrt(mean(a.map(x => (x - m) ** 2))); }

/* std-dev of the pitch track after discarding octave-jump outliers */
function pitchStd(freqs) {
  if (freqs.length < 2) return null;
  const med = median(freqs);
  const kept = freqs.filter(f => f > med * 0.75 && f < med * 1.33);
  return kept.length >= 2 ? stdev(kept) : null;
}

function summarizeWindow(w, hasCam, hasMic) {
  const MIN_PITCH_SAMPLES = 10;
  return {
    motion_mean: hasCam && w.motions.length >= 2 ? mean(w.motions) : null,
    pitch_std: hasMic && w.pitches.length >= MIN_PITCH_SAMPLES ? pitchStd(w.pitches) : null,
    voiced_ratio: hasMic && w.loudFrames > 0 ? w.pitches.length / w.loudFrames : 0,
  };
}

class SensorSession {
  constructor() {
    this.stream = null; this.video = null; this.analyser = null; this.audioCtx = null;
    this.timer = null; this.prev = null; this.win = null; this.onTick = null;
    this.hasCam = false; this.hasMic = false;
    this.canvas = document.createElement('canvas'); this.canvas.width = 64; this.canvas.height = 48;
    this.ctx = this.canvas.getContext ? this.canvas.getContext('2d', { willReadFrequently: true }) : null;
  }

  /* ask the browser for camera/mic; throws with a friendly message on failure */
  async open(wantCam, wantMic) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('This browser cannot access camera/microphone. Use Microsoft Edge or Google Chrome on http://localhost.');
    }
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: wantCam ? { width: 640, height: 480, facingMode: 'user' } : false,
        audio: wantMic ? { echoCancellation: true, noiseSuppression: false } : false,
      });
    } catch (e) {
      const hints = {
        NotAllowedError: 'Permission was blocked. Click the camera/lock icon in the address bar → Allow, then try again.',
        NotFoundError: 'No camera/microphone was found on this computer.',
        NotReadableError: 'The camera/microphone is being used by another app (Zoom, Teams, Camera app). Close it and try again.',
        OverconstrainedError: 'The camera does not support the requested settings.',
      };
      throw new Error(hints[e.name] || `Could not start camera/microphone (${e.name || e}).`);
    }
    this.hasCam = this.stream.getVideoTracks().length > 0;
    this.hasMic = this.stream.getAudioTracks().length > 0;
    if (this.hasMic) {
      const AC = window.AudioContext || window.webkitAudioContext;
      this.audioCtx = new AC();
      const src = this.audioCtx.createMediaStreamSource(this.stream);
      this.analyser = this.audioCtx.createAnalyser(); this.analyser.fftSize = 2048;
      src.connect(this.analyser);
    }
    return this;
  }

  attachVideo(videoEl) {
    this.video = videoEl;
    if (this.hasCam && videoEl) { videoEl.srcObject = this.stream; videoEl.play().catch(() => {}); }
  }

  start(onTick) {
    this.onTick = onTick; this.prev = null;
    if (this.timer) clearInterval(this.timer);
    this.timer = setInterval(() => this.tick(), 150);
  }
  stop() { if (this.timer) clearInterval(this.timer); this.timer = null; }
  close() {
    this.stop();
    if (this.stream) this.stream.getTracks().forEach(t => t.stop());
    if (this.audioCtx) this.audioCtx.close().catch(() => {});
    this.stream = null; this.hasCam = this.hasMic = false;
  }

  beginWindow() { this.win = { motions: [], pitches: [], loudFrames: 0 }; }
  endWindow() { const w = this.win; this.win = null; return w ? summarizeWindow(w, this.hasCam, this.hasMic) : {}; }

  tick() {
    const live = { motion: null, level: 0, pitch: null };
    if (this.hasCam && this.ctx && this.video && this.video.readyState >= 2 && this.video.videoWidth) {
      const vw = this.video.videoWidth, vh = this.video.videoHeight;      // centre 60 % crop: where the face usually is
      this.ctx.drawImage(this.video, vw * 0.2, vh * 0.2, vw * 0.6, vh * 0.6, 0, 0, 64, 48);
      const px = this.ctx.getImageData(0, 0, 64, 48).data, g = new Float32Array(64 * 48);
      for (let i = 0; i < g.length; i++) g[i] = 0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2];
      if (this.prev) { live.motion = frameMotion(this.prev, g); if (this.win) this.win.motions.push(live.motion); }
      this.prev = g;
    }
    if (this.hasMic && this.analyser) {
      const buf = new Float32Array(this.analyser.fftSize);
      this.analyser.getFloatTimeDomainData(buf);
      live.level = rms(buf);
      if (live.level > 0.02) {                                             // someone is speaking
        if (this.win) this.win.loudFrames++;
        const p = estimatePitch(buf, this.audioCtx.sampleRate);
        if (p) { live.pitch = p.freq; if (this.win) this.win.pitches.push(p.freq); }
      }
    }
    if (this.onTick) this.onTick(live);
  }
}

if (typeof module !== 'undefined') module.exports = { rms, estimatePitch, frameMotion, pitchStd, summarizeWindow, mean, median, stdev };
