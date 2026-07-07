/**
 * Interactive project demos — heuijee.github.io
 * Strict monochrome rendering. Each demo animates only while visible.
 */
(function () {
    'use strict';

    const C = {
        bg: '#0a0a0a',
        grid: 'rgba(255,255,255,0.05)',
        text: '#f2f2f2',
        muted: '#9c9c9c',
        faint: '#5f5f5f',
        white: '#ffffff',
        gray: '#c9c9c9',
        dim: '#8a8a8a',
        mono: '13px "JetBrains Mono", Consolas, monospace',
        monoSmall: '11px "JetBrains Mono", Consolas, monospace'
    };

    /* ---------- helpers ---------- */

    function setupCanvas(canvas) {
        const ctx = canvas.getContext('2d');
        let w = 0, h = 0;
        function resize() {
            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            w = canvas.clientWidth;
            h = canvas.clientHeight;
            canvas.width = Math.round(w * dpr);
            canvas.height = Math.round(h * dpr);
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        resize();
        window.addEventListener('resize', resize);
        return { ctx, size: () => ({ w, h }) };
    }

    function animateWhenVisible(canvas, tick) {
        let running = false, rafId = null, last = 0, t = 0;
        function loop(now) {
            if (!running) return;
            const dt = Math.min((now - last) / 1000, 0.05);
            last = now;
            t += dt;
            tick(dt, t);
            rafId = requestAnimationFrame(loop);
        }
        const io = new IntersectionObserver((entries) => {
            entries.forEach(e => {
                if (e.isIntersecting && !running) {
                    running = true;
                    last = performance.now();
                    rafId = requestAnimationFrame(loop);
                } else if (!e.isIntersecting && running) {
                    running = false;
                    if (rafId) cancelAnimationFrame(rafId);
                }
            });
        }, { threshold: 0.05 });
        io.observe(canvas);
    }

    function clear(ctx, w, h) {
        ctx.fillStyle = C.bg;
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = C.grid;
        ctx.lineWidth = 1;
        const step = 28;
        ctx.beginPath();
        for (let x = step; x < w; x += step) { ctx.moveTo(x, 0); ctx.lineTo(x, h); }
        for (let y = step; y < h; y += step) { ctx.moveTo(0, y); ctx.lineTo(w, y); }
        ctx.stroke();
    }

    function roundRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.arcTo(x + w, y, x + w, y + h, r);
        ctx.arcTo(x + w, y + h, x, y + h, r);
        ctx.arcTo(x, y + h, x, y, r);
        ctx.arcTo(x, y, x + w, y, r);
        ctx.closePath();
    }

    /* ======================================================================
       1. Sparsity-aware transformer accelerator — zero-skip systolic array
       ====================================================================== */
    (function sparsityDemo() {
        const canvas = document.getElementById('demo-sparsity');
        if (!canvas) return;
        const { ctx, size } = setupCanvas(canvas);
        const slider = document.getElementById('sparsity-slider');
        const valueEl = document.getElementById('sparsity-value');
        const statsEl = document.getElementById('sparsity-stats');

        const COLS = 12, ROWS = 6;
        let sparsity = slider ? slider.value / 100 : 0.6;
        let mask = [];

        function regenMask() {
            mask = [];
            for (let r = 0; r < ROWS; r++) {
                mask.push([]);
                for (let c = 0; c < COLS; c++) mask[r].push(Math.random() < sparsity);
            }
        }
        regenMask();

        if (slider) slider.addEventListener('input', () => {
            sparsity = slider.value / 100;
            valueEl.textContent = slider.value + '%';
            regenMask();
        });

        animateWhenVisible(canvas, (dt, t) => {
            const { w, h } = size();
            clear(ctx, w, h);

            const pad = 18;
            const gridW = w * 0.62;
            const cellW = (gridW - pad * 2) / COLS;
            const cellH = (h - pad * 2 - 26) / ROWS;
            const cell = Math.min(cellW, cellH);
            const ox = pad, oy = pad + 20;

            ctx.font = C.monoSmall;
            ctx.fillStyle = C.faint;
            ctx.fillText('SYSTOLIC MAC ARRAY (ZERO-SKIP)', ox, pad + 6);

            const wave = (t * 6) % (COLS + ROWS + 4);

            for (let r = 0; r < ROWS; r++) {
                for (let c = 0; c < COLS; c++) {
                    const x = ox + c * cell, y = oy + r * cell;
                    const d = c + r;
                    const inWave = Math.abs(d - wave) < 1.6;
                    const isZero = mask[r][c];
                    if (isZero) {
                        ctx.fillStyle = 'rgba(255,255,255,0.03)';
                        roundRect(ctx, x + 2, y + 2, cell - 4, cell - 4, 2);
                        ctx.fill();
                        ctx.strokeStyle = 'rgba(255,255,255,0.12)';
                        ctx.stroke();
                        ctx.fillStyle = 'rgba(255,255,255,0.25)';
                        ctx.font = '10px monospace';
                        ctx.fillText('×', x + cell / 2 - 3, y + cell / 2 + 3);
                    } else {
                        const glow = inWave ? 1 : 0.18;
                        ctx.fillStyle = `rgba(255,255,255,${0.08 + glow * 0.6})`;
                        roundRect(ctx, x + 2, y + 2, cell - 4, cell - 4, 2);
                        ctx.fill();
                        if (inWave) {
                            ctx.shadowColor = C.white;
                            ctx.shadowBlur = 10;
                            ctx.strokeStyle = C.white;
                            ctx.stroke();
                            ctx.shadowBlur = 0;
                        }
                    }
                }
            }

            // power bar + clock gate
            const px = ox + COLS * cell + 26;
            const pw = w - px - pad;
            if (pw > 90) {
                ctx.font = C.monoSmall;
                ctx.fillStyle = C.faint;
                ctx.fillText('DYNAMIC POWER', px, pad + 6);

                const power = 1 - sparsity * 0.82;
                const barH = h - pad * 2 - 48;
                const barY = oy + 6;
                ctx.strokeStyle = 'rgba(255,255,255,0.3)';
                ctx.strokeRect(px, barY, 26, barH);
                const fillH = barH * power;
                ctx.fillStyle = C.white;
                ctx.fillRect(px + 1, barY + barH - fillH, 24, fillH);
                ctx.fillStyle = C.text;
                ctx.font = C.mono;
                ctx.fillText(Math.round(power * 100) + '%', px + 34, barY + barH - fillH + 4);

                const clkPeriod = 0.5 + sparsity * 1.2;
                const on = (t % clkPeriod) < clkPeriod / 2;
                ctx.fillStyle = on ? C.white : 'rgba(255,255,255,0.12)';
                ctx.beginPath();
                ctx.arc(px + 8, h - pad - 6, 5, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = C.muted;
                ctx.font = C.monoSmall;
                ctx.fillText('RISC-V CLK GATE', px + 20, h - pad - 2);
            }

            if (statsEl) statsEl.textContent =
                'MACs skipped: ' + Math.round(sparsity * 100) + '% · est. power ▼' + Math.round(sparsity * 82) + '%';
        });
    })();

    /* ======================================================================
       2. HAB-1 — host ↔ bridge ↔ accelerator packet flow
       ====================================================================== */
    (function habDemo() {
        const canvas = document.getElementById('demo-hab');
        if (!canvas) return;
        const { ctx, size } = setupCanvas(canvas);
        const btn = document.getElementById('hab-reconfig');
        const statsEl = document.getElementById('hab-stats');

        const CHANNEL_OPTS = [2, 4, 8];
        let chIdx = 1;
        let packets = [];
        let reconfigFlash = 0;

        if (btn) btn.addEventListener('click', () => {
            chIdx = (chIdx + 1) % CHANNEL_OPTS.length;
            packets = [];
            reconfigFlash = 1;
        });

        function drawBlock(x, y, w, h, label, sub, flash) {
            ctx.fillStyle = 'rgba(255,255,255,0.05)';
            roundRect(ctx, x, y, w, h, 4);
            ctx.fill();
            ctx.strokeStyle = flash ? C.white : 'rgba(255,255,255,0.4)';
            ctx.lineWidth = flash ? 2 : 1.2;
            if (flash) { ctx.shadowColor = C.white; ctx.shadowBlur = 14; }
            ctx.stroke();
            ctx.shadowBlur = 0;
            ctx.lineWidth = 1;
            ctx.fillStyle = C.text;
            ctx.font = 'bold ' + C.mono;
            ctx.textAlign = 'center';
            ctx.fillText(label, x + w / 2, y + h / 2 - 4);
            ctx.fillStyle = C.faint;
            ctx.font = C.monoSmall;
            ctx.fillText(sub, x + w / 2, y + h / 2 + 14);
            ctx.textAlign = 'left';
        }

        animateWhenVisible(canvas, (dt, t) => {
            const { w, h } = size();
            clear(ctx, w, h);
            const channels = CHANNEL_OPTS[chIdx];
            reconfigFlash = Math.max(0, reconfigFlash - dt * 2);

            const bw = Math.min(150, w * 0.2), bh = h * 0.52;
            const by = (h - bh) / 2;
            const hostX = 20, bridgeX = (w - bw) / 2, accX = w - bw - 20;

            const laneGap = bh / (channels + 1);
            for (let seg = 0; seg < 2; seg++) {
                const x1 = seg === 0 ? hostX + bw : bridgeX + bw;
                const x2 = seg === 0 ? bridgeX : accX;
                for (let l = 0; l < channels; l++) {
                    const y = by + laneGap * (l + 1);
                    ctx.strokeStyle = 'rgba(255,255,255,0.14)';
                    ctx.beginPath();
                    ctx.moveTo(x1, y);
                    ctx.lineTo(x2, y);
                    ctx.stroke();
                }
            }

            if (Math.random() < dt * channels * 4) {
                packets.push({ seg: 0, lane: Math.floor(Math.random() * channels), p: 0, speed: 0.9 + Math.random() * 0.5 });
            }

            packets = packets.filter(pk => {
                pk.p += dt * pk.speed;
                if (pk.p >= 1) {
                    if (pk.seg === 0) { pk.seg = 1; pk.p = 0; pk.lane = Math.floor(Math.random() * channels); return true; }
                    return false;
                }
                return true;
            });
            packets.forEach(pk => {
                const x1 = pk.seg === 0 ? hostX + bw : bridgeX + bw;
                const x2 = pk.seg === 0 ? bridgeX : accX;
                const y = by + laneGap * (pk.lane + 1);
                const x = x1 + (x2 - x1) * pk.p;
                ctx.fillStyle = pk.seg === 0 ? C.dim : C.white;
                ctx.shadowColor = ctx.fillStyle;
                ctx.shadowBlur = 8;
                roundRect(ctx, x - 7, y - 3.5, 14, 7, 2);
                ctx.fill();
                ctx.shadowBlur = 0;
            });

            drawBlock(hostX, by, bw, bh, 'HOST CPU', 'RISC PIPELINE', false);
            drawBlock(bridgeX, by, bw, bh, 'HAB-1', channels + '-CH MPI BRIDGE', reconfigFlash > 0);
            drawBlock(accX, by, bw, bh, 'CNN ACCEL', 'RECONFIGURABLE', false);

            if (statsEl) statsEl.textContent = channels + ' channels · throughput ×' + (channels / 2).toFixed(1);
        });
    })();

    /* ======================================================================
       3. ARMuP — scalar vs µSIMD race
       ====================================================================== */
    (function simdDemo() {
        const canvas = document.getElementById('demo-simd');
        if (!canvas) return;
        const { ctx, size } = setupCanvas(canvas);
        const btn = document.getElementById('simd-replay');
        const statsEl = document.getElementById('simd-stats');

        const N = 32;
        let scalarDone = 0, simdDone = 0, cycScalar = 0, cycSimd = 0;
        let acc = 0, finishedAt = null;

        function reset() { scalarDone = 0; simdDone = 0; cycScalar = 0; cycSimd = 0; acc = 0; finishedAt = null; }
        if (btn) btn.addEventListener('click', reset);

        animateWhenVisible(canvas, (dt, t) => {
            const { w, h } = size();
            clear(ctx, w, h);

            acc += dt * 10;
            while (acc >= 1) {
                acc -= 1;
                if (scalarDone < N) { scalarDone += 1; cycScalar++; }
                if (simdDone < N) { simdDone += 4; cycSimd++; if (simdDone > N) simdDone = N; }
                if (scalarDone >= N && simdDone >= N && finishedAt === null) finishedAt = t;
            }
            if (finishedAt !== null && t - finishedAt > 2.2) reset();

            const pad = 20, labelW = 130;
            const trackW = w - pad * 2 - labelW;
            const cellW = trackW / N;
            const rows = [
                { y: h * 0.32, done: scalarDone, cyc: cycScalar, label: 'CORTEX-M0+', sub: '1 ELEM / CYCLE', color: C.dim },
                { y: h * 0.72, done: simdDone, cyc: cycSimd, label: 'ARMuP µSIMD', sub: '4 ELEMS / CYCLE', color: C.white }
            ];

            ctx.font = C.monoSmall;
            ctx.fillStyle = C.faint;
            ctx.fillText('WORKLOAD: 32-ELEMENT MAC LOOP (CNN LAYER)', pad, 18);

            rows.forEach(row => {
                ctx.fillStyle = C.text;
                ctx.font = 'bold ' + C.monoSmall;
                ctx.fillText(row.label, pad, row.y - 8);
                ctx.fillStyle = C.faint;
                ctx.font = C.monoSmall;
                ctx.fillText(row.sub, pad, row.y + 6);

                for (let i = 0; i < N; i++) {
                    const x = pad + labelW + i * cellW;
                    const filled = i < row.done;
                    ctx.fillStyle = filled ? row.color : 'rgba(255,255,255,0.06)';
                    if (filled && i >= row.done - 4) {
                        ctx.shadowColor = row.color;
                        ctx.shadowBlur = 6;
                    }
                    roundRect(ctx, x + 1, row.y - 14, cellW - 3, 24, 2);
                    ctx.fill();
                    ctx.shadowBlur = 0;
                }
                ctx.fillStyle = C.muted;
                ctx.font = C.mono;
                ctx.textAlign = 'right';
                ctx.fillText(row.cyc + ' cyc', w - pad, row.y + 4);
                ctx.textAlign = 'left';
            });

            if (simdDone >= N && scalarDone < N) {
                ctx.fillStyle = C.white;
                ctx.font = 'bold ' + C.mono;
                ctx.fillText('µSIMD FINISHED — SCALAR STILL WORKING…', pad + labelW, h - 8);
            } else if (finishedAt !== null) {
                ctx.fillStyle = C.white;
                ctx.font = 'bold ' + C.mono;
                ctx.fillText('µSIMD: ' + (cycScalar / cycSimd).toFixed(1) + '× FEWER CYCLES', pad + labelW, h - 8);
            }

            if (statsEl) statsEl.textContent = 'scalar ' + cycScalar + ' cyc vs µSIMD ' + cycSimd + ' cyc';
        });
    })();

    /* ======================================================================
       4. SNN CIM — spiking crossbar
       ====================================================================== */
    (function snnDemo() {
        const canvas = document.getElementById('demo-snn');
        if (!canvas) return;
        const { ctx, size } = setupCanvas(canvas);
        const btn = document.getElementById('snn-inject');
        const statsEl = document.getElementById('snn-stats');

        const IN = 6, OUT = 8;
        const weights = [];
        for (let i = 0; i < IN; i++) {
            weights.push([]);
            for (let j = 0; j < OUT; j++) weights[i].push(0.15 + Math.random() * 0.3);
        }
        let spikes = [];
        let potentials = new Array(OUT).fill(0);
        let flashes = new Array(OUT).fill(0);
        let fired = 0;
        const THRESH = 1.0;

        function inject(burst) {
            for (let i = 0; i < IN; i++) {
                if (Math.random() < (burst ? 0.95 : 0.4)) spikes.push({ row: i, p: -Math.random() * 0.15 });
            }
        }
        if (btn) btn.addEventListener('click', () => inject(true));
        canvas.addEventListener('pointerdown', () => inject(true));

        let spawnAcc = 0;

        animateWhenVisible(canvas, (dt, t) => {
            const { w, h } = size();
            clear(ctx, w, h);

            const padL = 90, padR = 26, padT = 30, barZone = 56;
            const gridW = w - padL - padR;
            const gridH = h - padT - barZone - 24;
            const rowY = i => padT + gridH * (i + 0.5) / IN;
            const colX = j => padL + gridW * (j + 0.5) / OUT;

            spawnAcc += dt;
            if (spawnAcc > 0.7) { spawnAcc = 0; inject(false); }

            ctx.font = C.monoSmall;
            ctx.fillStyle = C.faint;
            ctx.fillText('RRAM CROSSBAR', padL, 18);
            ctx.fillText('IF NEURONS', padL, h - 8);

            for (let i = 0; i < IN; i++) {
                ctx.strokeStyle = 'rgba(255,255,255,0.15)';
                ctx.beginPath();
                ctx.moveTo(padL - 40, rowY(i));
                ctx.lineTo(padL + gridW, rowY(i));
                ctx.stroke();
                ctx.fillStyle = C.faint;
                ctx.font = C.monoSmall;
                ctx.fillText('in' + i, padL - 70, rowY(i) + 4);
            }
            for (let j = 0; j < OUT; j++) {
                ctx.strokeStyle = 'rgba(255,255,255,0.15)';
                ctx.beginPath();
                ctx.moveTo(colX(j), padT);
                ctx.lineTo(colX(j), padT + gridH + 12);
                ctx.stroke();
            }
            for (let i = 0; i < IN; i++) {
                for (let j = 0; j < OUT; j++) {
                    ctx.fillStyle = `rgba(255,255,255,${0.10 + weights[i][j] * 0.7})`;
                    ctx.beginPath();
                    ctx.arc(colX(j), rowY(i), 2.6, 0, Math.PI * 2);
                    ctx.fill();
                }
            }

            spikes = spikes.filter(s => {
                const prev = s.p;
                s.p += dt * 0.85;
                for (let j = 0; j < OUT; j++) {
                    const cross = (j + 0.5) / OUT;
                    if (prev < cross && s.p >= cross) {
                        potentials[j] += weights[s.row][j] * 0.35;
                    }
                }
                if (s.p >= 1.05) return false;
                if (s.p >= 0) {
                    const x = padL - 40 + (gridW + 40) * s.p;
                    ctx.fillStyle = C.white;
                    ctx.shadowColor = C.white;
                    ctx.shadowBlur = 10;
                    ctx.beginPath();
                    ctx.arc(x, rowY(s.row), 4, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.shadowBlur = 0;
                }
                return true;
            });

            const barY = padT + gridH + 20;
            for (let j = 0; j < OUT; j++) {
                potentials[j] = Math.max(0, potentials[j] - dt * 0.12);
                if (potentials[j] >= THRESH) {
                    potentials[j] = 0;
                    flashes[j] = 1;
                    fired++;
                }
                flashes[j] = Math.max(0, flashes[j] - dt * 3);

                const bw = Math.min(26, gridW / OUT - 10);
                const x = colX(j) - bw / 2;
                const bh = 26;
                ctx.strokeStyle = 'rgba(255,255,255,0.3)';
                ctx.strokeRect(x, barY, bw, bh);
                const lvl = Math.min(potentials[j] / THRESH, 1);
                ctx.fillStyle = C.gray;
                ctx.fillRect(x + 1, barY + bh - lvl * (bh - 2) - 1, bw - 2, lvl * (bh - 2));
                if (flashes[j] > 0) {
                    ctx.fillStyle = `rgba(255,255,255,${flashes[j]})`;
                    ctx.shadowColor = C.white;
                    ctx.shadowBlur = 16;
                    ctx.beginPath();
                    ctx.arc(colX(j), barY + bh + 9, 5, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.shadowBlur = 0;
                }
            }

            if (statsEl) statsEl.textContent = 'output spikes: ' + fired + ' · threshold ' + THRESH.toFixed(1) + 'V';
        });
    })();

    /* ======================================================================
       5. GTA5 — lane detection + object detection overlay
       ====================================================================== */
    (function laneDemo() {
        const canvas = document.getElementById('demo-lane');
        if (!canvas) return;
        const { ctx, size } = setupCanvas(canvas);
        const btn = document.getElementById('lane-toggle');
        const statsEl = document.getElementById('lane-stats');

        let detect = true;
        if (btn) btn.addEventListener('click', () => { detect = !detect; });

        animateWhenVisible(canvas, (dt, t) => {
            const { w, h } = size();
            const sky = ctx.createLinearGradient(0, 0, 0, h * 0.45);
            sky.addColorStop(0, '#101010');
            sky.addColorStop(1, '#1c1c1c');
            ctx.fillStyle = sky;
            ctx.fillRect(0, 0, w, h * 0.45);
            ctx.fillStyle = '#0d0d0d';
            ctx.fillRect(0, h * 0.45, w, h * 0.55);

            const horizon = h * 0.45;
            const cx = w / 2;
            const roadHalfBottom = w * 0.34, roadHalfTop = w * 0.025;

            ctx.fillStyle = '#181818';
            ctx.beginPath();
            ctx.moveTo(cx - roadHalfTop, horizon);
            ctx.lineTo(cx + roadHalfTop, horizon);
            ctx.lineTo(cx + roadHalfBottom, h);
            ctx.lineTo(cx - roadHalfBottom, h);
            ctx.closePath();
            ctx.fill();

            ctx.strokeStyle = 'rgba(255,255,255,0.3)';
            ctx.lineWidth = 2;
            [[-1], [1]].forEach(([s]) => {
                ctx.beginPath();
                ctx.moveTo(cx + s * roadHalfTop, horizon);
                ctx.lineTo(cx + s * roadHalfBottom, h);
                ctx.stroke();
            });

            const phase = (t * 0.9) % 1;
            ctx.strokeStyle = 'rgba(255,255,255,0.45)';
            ctx.lineWidth = 3;
            for (let k = 0; k < 8; k++) {
                let a = (k + phase) / 8;
                let b = a + 0.045;
                a = a * a; b = b * b;
                if (b > 1) continue;
                const y1 = horizon + a * (h - horizon);
                const y2 = horizon + b * (h - horizon);
                ctx.beginPath();
                ctx.moveTo(cx, y1);
                ctx.lineTo(cx, y2);
                ctx.stroke();
            }

            const sway = Math.sin(t * 0.7) * w * 0.03;
            const carScale = 0.62 + Math.sin(t * 0.35) * 0.05;
            const cw = w * 0.11 * carScale, ch2 = cw * 0.62;
            const carX = cx + sway - cw / 2;
            const carY = horizon + (h - horizon) * 0.34 * carScale - ch2 / 2;
            ctx.fillStyle = '#2e2e2e';
            roundRect(ctx, carX, carY, cw, ch2, 4);
            ctx.fill();
            ctx.fillStyle = '#3d3d3d';
            roundRect(ctx, carX + cw * 0.14, carY - ch2 * 0.32, cw * 0.72, ch2 * 0.42, 3);
            ctx.fill();
            ctx.fillStyle = '#8a8a8a';
            ctx.fillRect(carX + 3, carY + ch2 * 0.32, cw * 0.13, 4);
            ctx.fillRect(carX + cw - cw * 0.13 - 3, carY + ch2 * 0.32, cw * 0.13, 4);

            if (detect) {
                const jit = () => (Math.random() - 0.5) * 1.6;
                ctx.strokeStyle = C.white;
                ctx.lineWidth = 3;
                ctx.shadowColor = C.white;
                ctx.shadowBlur = 8;
                [[-1], [1]].forEach(([s]) => {
                    ctx.beginPath();
                    ctx.moveTo(cx + s * roadHalfTop * 1.7 + jit(), horizon + 6);
                    ctx.lineTo(cx + s * roadHalfBottom * 0.86 + jit(), h - 4);
                    ctx.stroke();
                });
                ctx.shadowBlur = 0;

                ctx.fillStyle = 'rgba(255,255,255,0.05)';
                ctx.beginPath();
                ctx.moveTo(cx - roadHalfTop * 1.7, horizon + 6);
                ctx.lineTo(cx + roadHalfTop * 1.7, horizon + 6);
                ctx.lineTo(cx + roadHalfBottom * 0.86, h);
                ctx.lineTo(cx - roadHalfBottom * 0.86, h);
                ctx.closePath();
                ctx.fill();

                const conf = (0.91 + Math.sin(t * 2.1) * 0.04).toFixed(2);
                ctx.strokeStyle = C.white;
                ctx.lineWidth = 1.4;
                ctx.setLineDash([5, 4]);
                ctx.strokeRect(carX - 5, carY - ch2 * 0.42 - 5, cw + 10, ch2 * 1.5 + 10);
                ctx.setLineDash([]);
                ctx.fillStyle = C.white;
                ctx.font = C.monoSmall;
                ctx.fillText('CAR ' + conf, carX - 5, carY - ch2 * 0.42 - 10);

                ctx.fillStyle = C.white;
                ctx.font = C.monoSmall;
                ctx.fillText('CANNY + HOUGH · YOLOv4', 12, 20);
                ctx.fillText('LANE LOCK ✓', 12, 36);
            } else {
                ctx.fillStyle = C.faint;
                ctx.font = C.monoSmall;
                ctx.fillText('RAW CAMERA FEED (DETECTION OFF)', 12, 20);
            }

            if (statsEl) statsEl.textContent = detect ? 'detection: ON · ~2.7 FPS on Jetson-class HW' : 'detection: OFF';
        });
    })();

    /* ======================================================================
       6. Mini AI — local LLM terminal
       ====================================================================== */
    (function miniAiDemo() {
        const canvas = document.getElementById('demo-miniai');
        if (!canvas) return;
        const { ctx, size } = setupCanvas(canvas);
        const btn = document.getElementById('miniai-next');
        const statsEl = document.getElementById('miniai-stats');

        const CONVOS = [
            { q: 'who are you?', a: 'I am Mini-AI, a quantized LLM living on a tiny server in Heuijee\'s room. No cloud. No API keys. Just local silicon.' },
            { q: 'what hardware do you run on?', a: 'A compact low-power board with just enough RAM for a 4-bit quantized model. Heuijee tuned me to fit — pruning and quantization are kind of her thing.' },
            { q: 'is my data safe with you?', a: 'Everything stays inside this house. Your prompts never leave the local network — I literally have no internet route out.' },
            { q: 'do you know the Egg?', a: 'Of course. The egg-shaped speaker in Mom\'s room? I send it text, and it speaks with Heuijee\'s cloned voice. We are a team.' }
        ];

        let convoIdx = 0, phase = 0, charI = 0, charAcc = 0, pause = 0;
        const lines = [
            { pre: '$ ', text: './mini-ai --model q4_k --ctx 4096', color: C.muted },
            { pre: '', text: 'model loaded · 3.2 GB · local inference ready', color: C.faint }
        ];

        function next() { convoIdx = (convoIdx + 1) % CONVOS.length; phase = 0; charI = 0; pause = 0; }
        if (btn) btn.addEventListener('click', () => { if (phase >= 3) next(); else { phase = 3; } });

        animateWhenVisible(canvas, (dt, t) => {
            const { w, h } = size();
            ctx.fillStyle = '#070707';
            ctx.fillRect(0, 0, w, h);

            ctx.fillStyle = 'rgba(255,255,255,0.06)';
            ctx.fillRect(0, 0, w, 26);
            ['#5c5c5c', '#7a7a7a', '#9c9c9c'].forEach((c, i) => {
                ctx.fillStyle = c;
                ctx.beginPath();
                ctx.arc(16 + i * 18, 13, 5, 0, Math.PI * 2);
                ctx.fill();
            });
            ctx.fillStyle = C.faint;
            ctx.font = C.monoSmall;
            ctx.textAlign = 'center';
            ctx.fillText('mini-ai @ home-server — fully offline', w / 2, 17);
            ctx.textAlign = 'left';

            const convo = CONVOS[convoIdx];
            const speedQ = 22, speedA = 44;

            if (phase === 0) {
                charAcc += dt * speedQ;
                if (charAcc >= 1) { charI += Math.floor(charAcc); charAcc = 0; }
                if (charI >= convo.q.length) { charI = 0; phase = 1; pause = 0; }
            } else if (phase === 1) {
                pause += dt;
                if (pause > 0.9) { phase = 2; charI = 0; }
            } else if (phase === 2) {
                charAcc += dt * speedA;
                if (charAcc >= 1) { charI += Math.floor(charAcc); charAcc = 0; }
                if (charI >= convo.a.length) { phase = 3; pause = 0; }
            } else {
                pause += dt;
                if (pause > 4) next();
            }

            const qShown = phase === 0 ? convo.q.slice(0, charI) : convo.q;
            const aShown = phase < 2 ? '' : (phase === 2 ? convo.a.slice(0, charI) : convo.a);

            const pad = 16, lh = 19;
            let y = 26 + 24;
            ctx.font = C.mono;
            lines.forEach(l => {
                ctx.fillStyle = l.color;
                ctx.fillText(l.pre + l.text, pad, y);
                y += lh;
            });
            y += 6;

            ctx.fillStyle = C.dim;
            ctx.fillText('you> ', pad, y);
            ctx.fillStyle = C.text;
            ctx.fillText(qShown + (phase === 0 && Math.floor(t * 3) % 2 ? '▌' : ''), pad + 44, y);
            y += lh + 4;

            if (phase === 1) {
                ctx.fillStyle = C.faint;
                const dots = '.'.repeat(1 + Math.floor((t * 4) % 3));
                ctx.fillText('mini-ai> thinking' + dots, pad, y);
            } else if (phase >= 2) {
                ctx.fillStyle = C.white;
                ctx.fillText('mini-ai>', pad, y);
                ctx.fillStyle = C.muted;
                const maxW = w - pad * 2 - 78;
                const words = aShown.split(' ');
                let line = '', ly = y;
                words.forEach(word => {
                    const test = line ? line + ' ' + word : word;
                    if (ctx.measureText(test).width > maxW) {
                        ctx.fillText(line, pad + 78, ly);
                        ly += lh;
                        line = word;
                    } else line = test;
                });
                ctx.fillText(line + (phase === 2 && Math.floor(t * 3) % 2 ? '▌' : ''), pad + 78, ly);
            }

            if (statsEl) statsEl.textContent = phase >= 2
                ? '~' + (12 + Math.floor(Math.sin(t) * 2)) + ' tok/s · RAM 3.4/8 GB · cloud calls: 0'
                : 'cloud calls: 0 (always)';
        });
    })();

    /* ======================================================================
       7. CLEARSIGHT — AR glasses view: a holographic person reconstructed
          THROUGH the wall from sensor fusion, driven by a handheld remote.
       ====================================================================== */
    (function clearsightDemo() {
        const canvas = document.getElementById('demo-xray');
        if (!canvas) return;
        const { ctx, size } = setupCanvas(canvas);
        const btn = document.getElementById('xray-mode');
        const statsEl = document.getElementById('xray-stats');

        const HOLO = '143,171,186';        // soft cyan hologram
        const MODES = ['HOLOGRAM', 'SKELETON', 'X-RAY SCAN'];
        const RLBL = ['HOLO', 'SKEL', 'SCAN'];
        let mode = 0;
        function setBtn() { if (btn) btn.innerHTML = '<i class="fas fa-satellite-dish"></i> Remote &#9656; ' + MODES[mode]; }
        if (btn) { btn.addEventListener('click', () => { mode = (mode + 1) % MODES.length; setBtn(); }); setBtn(); }
        canvas.addEventListener('pointerdown', () => { mode = (mode + 1) % MODES.length; setBtn(); });

        // humanoid skeleton (normalized 0..1 inside the person box) with idle/walk cycle
        function joints(t) {
            const sway = Math.sin(t * 1.2) * 0.02;
            const breathe = Math.sin(t * 2) * 0.01;
            const armL = Math.sin(t * 1.6) * 0.13, armR = -Math.sin(t * 1.6) * 0.13;
            const legL = Math.sin(t * 1.6) * 0.10, legR = -Math.sin(t * 1.6) * 0.10;
            return {
                head: [0.5 + sway, 0.10], neck: [0.5 + sway, 0.22],
                sL: [0.40 + sway, 0.24], sR: [0.60 + sway, 0.24],
                eL: [0.34 + sway, 0.38 + armL * 0.3], eR: [0.66 + sway, 0.38 + armR * 0.3],
                hL: [0.30 + sway, 0.52 + armL], hR: [0.70 + sway, 0.52 + armR],
                hip: [0.5 + sway, 0.52 + breathe], pL: [0.44 + sway, 0.54], pR: [0.56 + sway, 0.54],
                kL: [0.42 + sway, 0.72 + legL * 0.2], kR: [0.58 + sway, 0.72 + legR * 0.2],
                fL: [0.40 + sway, 0.92 + legL], fR: [0.60 + sway, 0.92 + legR]
            };
        }
        const BONES = [['neck', 'head'], ['sL', 'sR'], ['neck', 'sL'], ['neck', 'sR'], ['sL', 'eL'], ['eL', 'hL'],
            ['sR', 'eR'], ['eR', 'hR'], ['neck', 'hip'], ['hip', 'pL'], ['hip', 'pR'], ['pL', 'kL'], ['kL', 'fL'], ['pR', 'kR'], ['kR', 'fR']];

        let box;
        function J(j, name) { const p = j[name]; return [box.x + p[0] * box.w, box.y + p[1] * box.h]; }

        function drawWall(w, h) {
            ctx.fillStyle = '#0b1018';
            ctx.fillRect(0, 0, w, h);
            ctx.strokeStyle = 'rgba(150,175,205,0.08)';
            ctx.lineWidth = 1;
            const bh = 30, bw = 76;
            for (let row = 0; row * bh < h + bh; row++) {
                const y = row * bh;
                ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
                const off = (row % 2) * (bw / 2);
                for (let x = off; x < w; x += bw) { ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + bh); ctx.stroke(); }
            }
        }

        function drawHologram(t, alpha, skeletonOnly) {
            const j = joints(t);
            const cx = box.x + box.w * 0.5, fy = box.y + box.h * 0.94;
            // holo emitter platform
            ctx.strokeStyle = 'rgba(' + HOLO + ',' + (0.4 * alpha) + ')';
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.ellipse(cx, fy, box.w * 0.42, box.w * 0.10, 0, 0, Math.PI * 2); ctx.stroke();
            // translucent silhouette
            if (!skeletonOnly) {
                ctx.fillStyle = 'rgba(' + HOLO + ',' + (0.10 * alpha) + ')';
                ctx.beginPath();
                ctx.moveTo.apply(ctx, J(j, 'sL')); ctx.lineTo.apply(ctx, J(j, 'hL')); ctx.lineTo.apply(ctx, J(j, 'eL'));
                ctx.lineTo.apply(ctx, J(j, 'pL')); ctx.lineTo.apply(ctx, J(j, 'fL')); ctx.lineTo.apply(ctx, J(j, 'fR'));
                ctx.lineTo.apply(ctx, J(j, 'pR')); ctx.lineTo.apply(ctx, J(j, 'eR')); ctx.lineTo.apply(ctx, J(j, 'hR'));
                ctx.lineTo.apply(ctx, J(j, 'sR')); ctx.closePath(); ctx.fill();
            }
            // bones
            ctx.strokeStyle = 'rgba(' + HOLO + ',' + (0.85 * alpha) + ')';
            ctx.lineWidth = 2;
            ctx.shadowColor = 'rgba(' + HOLO + ',0.8)'; ctx.shadowBlur = 8;
            BONES.forEach(b => { const a = J(j, b[0]), c = J(j, b[1]); ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(c[0], c[1]); ctx.stroke(); });
            const hd = J(j, 'head');
            ctx.beginPath(); ctx.arc(hd[0], hd[1], box.w * 0.10, 0, Math.PI * 2); ctx.stroke();
            ctx.shadowBlur = 0;
            // joints
            ctx.fillStyle = 'rgba(' + HOLO + ',' + alpha + ')';
            Object.keys(j).forEach(k => { const p = J(j, k); ctx.beginPath(); ctx.arc(p[0], p[1], 2.4, 0, Math.PI * 2); ctx.fill(); });
        }

        function scanlines(alpha) {
            ctx.fillStyle = 'rgba(11,16,24,' + (0.35 * alpha) + ')';
            for (let y = box.y; y < box.y + box.h; y += 4) ctx.fillRect(box.x - 12, y, box.w + 24, 2);
        }

        function drawRemote(w, h) {
            const rw = 54, rh = 100, rx = w - rw - 14, ry = h - rh - 28;
            ctx.fillStyle = 'rgba(20,28,42,0.92)';
            roundRect(ctx, rx, ry, rw, rh, 8); ctx.fill();
            ctx.strokeStyle = 'rgba(' + HOLO + ',0.5)'; ctx.lineWidth = 1; ctx.stroke();
            ctx.fillStyle = 'rgba(150,175,205,0.55)'; ctx.font = '7px "JetBrains Mono", monospace';
            ctx.fillText('REMOTE', rx + 8, ry + 13);
            for (let i = 0; i < MODES.length; i++) {
                const by = ry + 24 + i * 23, on = i === mode;
                ctx.fillStyle = on ? 'rgba(' + HOLO + ',0.9)' : 'rgba(150,175,205,0.15)';
                if (on) { ctx.shadowColor = 'rgba(' + HOLO + ',0.8)'; ctx.shadowBlur = 8; }
                roundRect(ctx, rx + 8, by, rw - 16, 16, 3); ctx.fill(); ctx.shadowBlur = 0;
                ctx.fillStyle = on ? '#0b1018' : 'rgba(150,175,205,0.5)'; ctx.font = '7px "JetBrains Mono", monospace';
                ctx.fillText(RLBL[i], rx + 13, by + 11);
            }
            // IR command beam toward the scene
            ctx.strokeStyle = 'rgba(' + HOLO + ',0.22)';
            ctx.setLineDash([4, 4]);
            ctx.beginPath(); ctx.moveTo(rx, ry + 8); ctx.lineTo(box.x + box.w * 0.7, box.y + box.h * 0.3); ctx.stroke();
            ctx.setLineDash([]);
        }

        animateWhenVisible(canvas, (dt, t) => {
            const { w, h } = size();
            drawWall(w, h);

            const ph = h * 0.74, pw = ph * 0.42;
            box = { x: w * 0.5 - pw / 2 - w * 0.05, y: h * 0.13, w: pw, h: ph };
            const m = MODES[mode];

            // lens "reveal" window — where the glasses see through the wall
            const rw = Math.min(w, h) * 0.62;
            const rcx = box.x + box.w * 0.5, rcy = box.y + box.h * 0.5;
            ctx.fillStyle = 'rgba(6,10,18,0.55)';
            roundRect(ctx, rcx - rw * 0.55, rcy - rw * 0.62, rw * 1.1, rw * 1.24, 18); ctx.fill();
            ctx.strokeStyle = 'rgba(' + HOLO + ',0.45)'; ctx.lineWidth = 1.5;
            roundRect(ctx, rcx - rw * 0.55, rcy - rw * 0.62, rw * 1.1, rw * 1.24, 18); ctx.stroke();

            if (m === 'X-RAY SCAN') {
                const sy = box.y + ((t * 0.4) % 1) * box.h;
                ctx.save();
                ctx.beginPath(); ctx.rect(0, box.y, w, sy - box.y); ctx.clip();
                drawHologram(t, 1, true);
                ctx.restore();
                drawHologram(t, 0.16, true);
                ctx.strokeStyle = 'rgba(' + HOLO + ',0.85)'; ctx.lineWidth = 2;
                ctx.shadowColor = 'rgba(' + HOLO + ',0.9)'; ctx.shadowBlur = 10;
                ctx.beginPath(); ctx.moveTo(box.x - 12, sy); ctx.lineTo(box.x + box.w + 12, sy); ctx.stroke(); ctx.shadowBlur = 0;
            } else {
                const flick = 0.82 + Math.sin(t * 20) * 0.06 + (Math.random() < 0.04 ? -0.2 : 0);
                drawHologram(t, flick, m === 'SKELETON');
                scanlines(flick);
            }

            // head reticle
            const hx = box.x + box.w * 0.5, hy = box.y + box.h * 0.10;
            ctx.strokeStyle = 'rgba(' + HOLO + ',0.7)'; ctx.lineWidth = 1;
            ctx.strokeRect(hx - 16, hy - 16, 32, 32);
            ctx.fillStyle = 'rgba(' + HOLO + ',0.9)'; ctx.font = C.monoSmall;
            ctx.fillText('HUMAN', hx - 16, hy - 20);

            // glasses HUD frame
            ctx.strokeStyle = 'rgba(150,175,205,0.4)'; ctx.lineWidth = 2;
            const mm = 12, L = 22;
            [[mm, mm, 1, 1], [w - mm, mm, -1, 1], [mm, h - mm, 1, -1], [w - mm, h - mm, -1, -1]].forEach(function (a) {
                ctx.beginPath(); ctx.moveTo(a[0] + a[2] * L, a[1]); ctx.lineTo(a[0], a[1]); ctx.lineTo(a[0], a[1] + a[3] * L); ctx.stroke();
            });

            const dist = (4.0 + Math.sin(t * 0.6) * 0.3).toFixed(1);
            ctx.font = C.monoSmall;
            ctx.fillStyle = 'rgba(' + HOLO + ',0.95)';
            ctx.fillText('CLEARSIGHT · THROUGH-WALL · ' + m, 20, 24);
            ctx.fillStyle = C.faint;
            ctx.fillText('HUMAN · ' + dist + ' m · behind wall · lock 0.' + (88 + Math.floor(Math.abs(Math.sin(t)) * 8)), 20, h - 22);

            drawRemote(w, h);

            if (statsEl) statsEl.textContent = 'holo reconstruction · 1 human · ' + dist + ' m · mode: ' + m;
        });
    })();

    /* ======================================================================
       8. EGG-01 PCB — board viewer with layer toggle
       ====================================================================== */
    (function pcbDemo() {
        const canvas = document.getElementById('demo-pcb');
        if (!canvas) return;
        const { ctx, size } = setupCanvas(canvas);
        const btn = document.getElementById('pcb-layer');
        const statsEl = document.getElementById('pcb-stats');

        const LAYERS = ['ALL LAYERS', 'COPPER (TOP)', 'SILKSCREEN', 'DRILL'];
        let layer = 0;
        if (btn) btn.addEventListener('click', () => { layer = (layer + 1) % LAYERS.length; });

        // board geometry in normalized coords (0..1)
        const parts = [
            { x: 0.38, y: 0.16, w: 0.24, h: 0.24, label: 'MCU', pins: 8 },
            { x: 0.10, y: 0.20, w: 0.16, h: 0.14, label: 'AMP', pins: 4 },
            { x: 0.72, y: 0.20, w: 0.16, h: 0.13, label: 'CHG', pins: 4 },
            { x: 0.12, y: 0.62, w: 0.13, h: 0.12, label: 'TOUCH', pins: 3 },
            { x: 0.70, y: 0.60, w: 0.19, h: 0.14, label: 'PWR', pins: 4 }
        ];
        const traces = [
            [0.26, 0.27, 0.38, 0.27], [0.26, 0.31, 0.33, 0.31, 0.33, 0.36, 0.38, 0.36],
            [0.62, 0.27, 0.72, 0.27], [0.62, 0.31, 0.67, 0.31, 0.67, 0.24, 0.72, 0.24],
            [0.50, 0.40, 0.50, 0.52, 0.25, 0.52, 0.25, 0.62],
            [0.55, 0.40, 0.55, 0.56, 0.70, 0.56, 0.70, 0.60],
            [0.44, 0.40, 0.44, 0.60, 0.30, 0.60, 0.30, 0.68, 0.25, 0.68],
            [0.80, 0.33, 0.80, 0.60],
            [0.18, 0.34, 0.18, 0.62]
        ];
        const drills = [
            [0.06, 0.08], [0.94, 0.08], [0.06, 0.90], [0.94, 0.90],
            [0.50, 0.80], [0.32, 0.80], [0.68, 0.80]
        ];
        const speakerPads = [];
        for (let r = 0; r < 3; r++)
            for (let k = 0; k < 5 - r; k++)
                speakerPads.push([0.40 + k * 0.05 + r * 0.025, 0.66 + r * 0.07]);

        animateWhenVisible(canvas, (dt, t) => {
            const { w, h } = size();
            ctx.fillStyle = '#070707';
            ctx.fillRect(0, 0, w, h);

            const m = 26;
            const bx = m, by = m, bw2 = w - m * 2, bh2 = h - m * 2;
            const X = v => bx + v * bw2;
            const Y = v => by + v * bh2;
            const showCopper = layer === 0 || layer === 1;
            const showSilk = layer === 0 || layer === 2;
            const showDrill = layer === 0 || layer === 3;

            // board outline (egg-ish rounded board)
            ctx.fillStyle = '#101010';
            roundRect(ctx, bx, by, bw2, bh2, 26);
            ctx.fill();
            ctx.strokeStyle = 'rgba(255,255,255,0.5)';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // copper traces
            if (showCopper) {
                ctx.strokeStyle = 'rgba(255,255,255,0.55)';
                ctx.lineWidth = 2;
                ctx.lineCap = 'round';
                traces.forEach(tr => {
                    ctx.beginPath();
                    ctx.moveTo(X(tr[0]), Y(tr[1]));
                    for (let i = 2; i < tr.length; i += 2) ctx.lineTo(X(tr[i]), Y(tr[i + 1]));
                    ctx.stroke();
                });
                // component pads + bodies
                parts.forEach(p => {
                    ctx.fillStyle = 'rgba(255,255,255,0.10)';
                    ctx.fillRect(X(p.x), Y(p.y), p.w * bw2, p.h * bh2);
                    ctx.strokeStyle = 'rgba(255,255,255,0.7)';
                    ctx.lineWidth = 1.2;
                    ctx.strokeRect(X(p.x), Y(p.y), p.w * bw2, p.h * bh2);
                    for (let i = 0; i < p.pins; i++) {
                        const frac = (i + 0.5) / p.pins;
                        ctx.fillStyle = 'rgba(255,255,255,0.85)';
                        ctx.fillRect(X(p.x) - 4, Y(p.y) + frac * p.h * bh2 - 2, 4, 4);
                        ctx.fillRect(X(p.x + p.w), Y(p.y) + frac * p.h * bh2 - 2, 4, 4);
                    }
                });
                // speaker pad grid
                speakerPads.forEach(sp => {
                    ctx.fillStyle = 'rgba(255,255,255,0.7)';
                    ctx.beginPath();
                    ctx.arc(X(sp[0]), Y(sp[1]), 3, 0, Math.PI * 2);
                    ctx.fill();
                });
            }

            // silkscreen
            if (showSilk) {
                ctx.fillStyle = 'rgba(255,255,255,0.9)';
                ctx.font = 'bold ' + C.monoSmall;
                parts.forEach(p => {
                    ctx.fillText(p.label, X(p.x) + 4, Y(p.y) - 5);
                });
                ctx.fillText('EGG-01 REV.B', X(0.35), Y(0.95));
                ctx.fillText('HJ-YUN © 2026', X(0.05), Y(0.06));
                ctx.font = C.monoSmall;
                ctx.fillStyle = 'rgba(255,255,255,0.5)';
                ctx.fillText('SPK', X(0.40), Y(0.63));
                // outline arrow
                ctx.strokeStyle = 'rgba(255,255,255,0.5)';
                ctx.setLineDash([4, 4]);
                ctx.strokeRect(X(0.37), Y(0.61), 0.28 * bw2, 0.22 * bh2);
                ctx.setLineDash([]);
            }

            // drill holes
            if (showDrill) {
                drills.forEach(d => {
                    ctx.fillStyle = '#070707';
                    ctx.beginPath();
                    ctx.arc(X(d[0]), Y(d[1]), 6, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.strokeStyle = 'rgba(255,255,255,0.8)';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(X(d[0]) - 9, Y(d[1])); ctx.lineTo(X(d[0]) + 9, Y(d[1]));
                    ctx.moveTo(X(d[0]), Y(d[1]) - 9); ctx.lineTo(X(d[0]), Y(d[1]) + 9);
                    ctx.stroke();
                });
            }

            // scanning beam sweep
            const sweepX = bx + ((t * 60) % (bw2 + 80)) - 40;
            const beam = ctx.createLinearGradient(sweepX - 30, 0, sweepX + 30, 0);
            beam.addColorStop(0, 'rgba(255,255,255,0)');
            beam.addColorStop(0.5, 'rgba(255,255,255,0.07)');
            beam.addColorStop(1, 'rgba(255,255,255,0)');
            ctx.fillStyle = beam;
            ctx.fillRect(sweepX - 30, by, 60, bh2);

            // HUD
            ctx.font = C.monoSmall;
            ctx.fillStyle = C.white;
            ctx.fillText('LAYER: ' + LAYERS[layer], 12, 18);

            if (statsEl) statsEl.textContent =
                LAYERS[layer].toLowerCase() + ' · 2-layer FR-4 · 5 components · 7 drills';
        });
    })();

    /* ======================================================================
       9. Egg — 3D device (Three.js) + Web Speech TTS (monochrome)
       ====================================================================== */
    (function eggDemo() {
        const canvas = document.getElementById('demo-egg');
        if (!canvas) return;
        const statusEl = document.getElementById('egg-status');
        const speakBtn = document.getElementById('egg-speak');

        if (typeof THREE === 'undefined') {
            const { ctx, size } = setupCanvas(canvas);
            const { w, h } = size();
            clear(ctx, w, h);
            ctx.fillStyle = C.muted;
            ctx.font = C.mono;
            ctx.textAlign = 'center';
            ctx.fillText('3D viewer unavailable (three.js failed to load)', w / 2, h / 2);
            return;
        }

        const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(38, 2, 0.1, 50);
        camera.position.set(0, 0.35, 4.6);

        scene.add(new THREE.AmbientLight(0xffffff, 0.5));
        const keyLight = new THREE.DirectionalLight(0xffffff, 0.95);
        keyLight.position.set(3, 4, 5);
        scene.add(keyLight);
        const rimLight = new THREE.DirectionalLight(0xffffff, 0.35);
        rimLight.position.set(-4, 1, -3);
        scene.add(rimLight);

        const group = new THREE.Group();
        scene.add(group);

        const pts = [];
        const SEGS = 40;
        for (let i = 0; i <= SEGS; i++) {
            const th = (i / SEGS) * Math.PI;
            const y = Math.cos(th) * 1.05;
            const r = Math.sin(th) * (0.78 - 0.14 * Math.cos(th));
            pts.push(new THREE.Vector2(Math.max(r, 0.0001), y));
        }
        const eggGeo = new THREE.LatheGeometry(pts, 48);
        const eggMat = new THREE.MeshStandardMaterial({ color: 0xeeeeea, roughness: 0.38, metalness: 0.02 });
        const egg = new THREE.Mesh(eggGeo, eggMat);
        group.add(egg);

        function radiusAt(y) {
            const c = Math.max(-1, Math.min(1, y / 1.05));
            const s = Math.sqrt(1 - c * c);
            return s * (0.78 - 0.14 * c);
        }

        const dotGeo = new THREE.SphereGeometry(0.024, 8, 8);
        const dotMat = new THREE.MeshStandardMaterial({ color: 0x1c1c1c, roughness: 0.6 });
        for (let row = 0; row < 5; row++) {
            const y = -0.42 + row * 0.13;
            const r = radiusAt(y);
            const nDots = 7 - Math.abs(row - 2);
            for (let k = 0; k < nDots; k++) {
                const phi = (k - (nDots - 1) / 2) * 0.16;
                const dot = new THREE.Mesh(dotGeo, dotMat);
                dot.position.set(Math.sin(phi) * r * 0.99, y, Math.cos(phi) * r * 0.99);
                group.add(dot);
            }
        }

        const ledMat = new THREE.MeshStandardMaterial({
            color: 0x1a1a1a, emissive: 0xffffff, emissiveIntensity: 0.3, roughness: 0.3
        });
        const led = new THREE.Mesh(new THREE.TorusGeometry(radiusAt(0.72) * 0.92, 0.028, 12, 48), ledMat);
        led.rotation.x = Math.PI / 2;
        led.position.y = 0.72;
        group.add(led);

        const base = new THREE.Mesh(
            new THREE.CylinderGeometry(0.62, 0.7, 0.14, 40),
            new THREE.MeshStandardMaterial({ color: 0x141414, roughness: 0.5, metalness: 0.3 })
        );
        base.position.y = -1.12;
        group.add(base);
        const baseGlow = new THREE.Mesh(
            new THREE.TorusGeometry(0.64, 0.012, 8, 48),
            new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.6 })
        );
        baseGlow.rotation.x = Math.PI / 2;
        baseGlow.position.y = -1.04;
        group.add(baseGlow);

        group.position.y = 0.1;

        let dragging = false, moved = 0, lastX = 0, lastY = 0;
        let rotY = -0.4, rotX = 0.08, autoSpin = true;

        canvas.addEventListener('pointerdown', e => {
            dragging = true; moved = 0; lastX = e.clientX; lastY = e.clientY;
            canvas.setPointerCapture(e.pointerId);
        });
        canvas.addEventListener('pointermove', e => {
            if (!dragging) return;
            const dx = e.clientX - lastX, dy = e.clientY - lastY;
            moved += Math.abs(dx) + Math.abs(dy);
            rotY += dx * 0.01;
            rotX = Math.max(-0.6, Math.min(0.6, rotX + dy * 0.006));
            lastX = e.clientX; lastY = e.clientY;
            autoSpin = false;
        });
        canvas.addEventListener('pointerup', () => {
            dragging = false;
            if (moved < 6) speak();
            setTimeout(() => { autoSpin = true; }, 2500);
        });

        const PHRASES = [
            '엄마, 오늘도 사랑해요.',
            '약 드실 시간이에요, 엄마!',
            '물 한 잔 마시고 가실래요?',
            '오늘 날씨 좋아요. 같이 산책 어때요?'
        ];
        let phraseIdx = 0, speaking = false;

        function speak() {
            if (!('speechSynthesis' in window)) {
                if (statusEl) statusEl.textContent = 'this browser has no speech synthesis';
                return;
            }
            if (speaking) return;
            const phrase = PHRASES[phraseIdx];
            phraseIdx = (phraseIdx + 1) % PHRASES.length;
            const u = new SpeechSynthesisUtterance(phrase);
            u.lang = 'ko-KR';
            const ko = speechSynthesis.getVoices().find(v => v.lang && v.lang.startsWith('ko'));
            if (ko) u.voice = ko;
            u.onstart = () => { speaking = true; if (statusEl) statusEl.textContent = '🔊 “' + phrase + '”'; };
            u.onend = () => { speaking = false; if (statusEl) statusEl.textContent = 'tap the egg or the button'; };
            u.onerror = () => { speaking = false; };
            speechSynthesis.cancel();
            speechSynthesis.speak(u);
        }
        if (speakBtn) speakBtn.addEventListener('click', speak);
        if ('speechSynthesis' in window) speechSynthesis.getVoices();

        function resize() {
            const w = canvas.clientWidth, h = canvas.clientHeight;
            if (canvas.width !== w || canvas.height !== h) {
                renderer.setSize(w, h, false);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
                camera.aspect = w / h;
                camera.updateProjectionMatrix();
            }
        }

        let running = false, rafId = null;
        function loop(now) {
            if (!running) return;
            resize();
            const t = now / 1000;
            if (autoSpin && !dragging) rotY += 0.004;
            group.rotation.y = rotY;
            group.rotation.x = rotX;

            if (speaking) {
                const s = 1 + Math.sin(t * 18) * 0.015;
                egg.scale.set(1 / s, s, 1 / s);
                ledMat.emissiveIntensity = 1.1 + Math.sin(t * 14) * 0.7;
            } else {
                egg.scale.set(1, 1, 1);
                ledMat.emissiveIntensity = 0.3 + Math.sin(t * 2) * 0.12;
            }
            baseGlow.material.opacity = 0.45 + Math.sin(t * 2.5) * 0.2;

            renderer.render(scene, camera);
            rafId = requestAnimationFrame(loop);
        }
        const io = new IntersectionObserver(entries => {
            entries.forEach(e => {
                if (e.isIntersecting && !running) { running = true; rafId = requestAnimationFrame(loop); }
                else if (!e.isIntersecting && running) { running = false; if (rafId) cancelAnimationFrame(rafId); }
            });
        }, { threshold: 0.05 });
        io.observe(canvas);
    })();

})();
