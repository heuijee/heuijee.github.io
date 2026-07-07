/**
 * Homepage hero — interactive "FRONTEND → MIDDLEWARE → HARDWARE" 3D stack.
 * Three floating layers, each a clickable research branch:
 *   top    · FRONTEND  (autonomous driving)  → Autonomy   (focus-auto)
 *   middle · MIDDLEWARE (framework)           → Memory     (focus-memory)
 *   bottom · HARDWARE  (silicon die)          → Core       (focus-core)
 * Signals flow downward through the stack. Drag to rotate, click a layer to
 * open its branch. Hovering a layer (or the left branch rows) highlights it.
 * three.js r147, soft low-glare palette.
 */
(function () {
    'use strict';

    const canvas = document.getElementById('hero3d');
    if (!canvas || typeof THREE === 'undefined') return;

    const CYAN = 0x8fabba, ROSE = 0xc19cae, GOLD = 0xcbb389;
    const cGold = new THREE.Color(GOLD), cRose = new THREE.Color(ROSE), cCyan = new THREE.Color(CYAN);
    const tmpA = new THREE.Color(), tmpB = new THREE.Color();
    function flowColor(f, out) {
        if (f < 0.5) { tmpA.copy(cGold); tmpB.copy(cRose); return out.copy(tmpA).lerp(tmpB, f / 0.5); }
        tmpA.copy(cRose); tmpB.copy(cCyan); return out.copy(tmpA).lerp(tmpB, (f - 0.5) / 0.5);
    }

    // click a layer → scroll to that section of the narrative
    const SCROLL = { auto: 'l-application', memory: 'l-memory', core: 'l-architecture' };

    const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    if ('outputEncoding' in renderer) renderer.outputEncoding = THREE.sRGBEncoding;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(36, 2, 0.1, 100);
    camera.position.set(0, 1.9, 5.0);
    camera.lookAt(0, -0.05, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.64));
    const key = new THREE.DirectionalLight(0xeaf4ff, 0.9); key.position.set(3, 6, 4); scene.add(key);
    const fill = new THREE.PointLight(0x9fb4d6, 0.55); fill.position.set(-4, 2, -2); scene.add(fill);

    const stack = new THREE.Group();
    scene.add(stack);

    const SLAB = 2.6, THICK = 0.12;
    const slabGeo = new THREE.BoxGeometry(SLAB, THICK, SLAB);
    function makeEdges(mesh, color) {
        const e = new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry),
            new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: 0.85 }));
        mesh.add(e); return e;
    }

    /* TOP — FRONTEND (autonomous driving, textured) */
    const topLayer = new THREE.Group(); stack.add(topLayer);
    const topMat = new THREE.MeshStandardMaterial({ color: 0x2a3444, roughness: 0.7, metalness: 0.1, emissive: 0x0a0f18, emissiveIntensity: 0.4 });
    const topSlab = new THREE.Mesh(slabGeo, topMat); topLayer.add(topSlab);
    const topEdge = makeEdges(topSlab, GOLD);
    new THREE.TextureLoader().load('images/Project3.png',
        function (tex) { if ('encoding' in tex) tex.encoding = THREE.sRGBEncoding; topMat.map = tex; topMat.color.set(0xffffff); topMat.needsUpdate = true; },
        undefined, function () {});

    /* MIDDLE — MIDDLEWARE (translucent framework plane + graph nodes) */
    const midLayer = new THREE.Group(); stack.add(midLayer);
    const midMat = new THREE.MeshStandardMaterial({ color: 0x151d2b, roughness: 0.5, metalness: 0.2, transparent: true, opacity: 0.4 });
    const midSlab = new THREE.Mesh(slabGeo, midMat); midLayer.add(midSlab);
    const midEdge = makeEdges(midSlab, ROSE);
    const gridMat = new THREE.LineBasicMaterial({ color: ROSE, transparent: true, opacity: 0.32 });
    const gpts = []; const GN = 5, half = SLAB / 2 - 0.2;
    for (let i = 0; i <= GN; i++) {
        const p = -half + (2 * half) * i / GN;
        gpts.push(new THREE.Vector3(-half, THICK / 2 + 0.01, p), new THREE.Vector3(half, THICK / 2 + 0.01, p));
        gpts.push(new THREE.Vector3(p, THICK / 2 + 0.01, -half), new THREE.Vector3(p, THICK / 2 + 0.01, half));
    }
    midLayer.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(gpts), gridMat));
    const nodeGeo = new THREE.SphereGeometry(0.07, 12, 12);
    const nodes = [];
    for (let i = 0; i < 6; i++) {
        const n = new THREE.Mesh(nodeGeo, new THREE.MeshStandardMaterial({ color: 0x2a1f2a, emissive: ROSE, emissiveIntensity: 0.8, roughness: 0.4 }));
        n.position.set(Math.sin(i * 2.3) * half * 0.7, THICK / 2 + 0.06, Math.cos(i * 1.7) * half * 0.7);
        n.userData.ph = i; midLayer.add(n); nodes.push(n);
    }

    /* BOTTOM — HARDWARE (silicon die + core grid) */
    const botLayer = new THREE.Group(); stack.add(botLayer);
    const botMat = new THREE.MeshStandardMaterial({ color: 0x121a28, roughness: 0.45, metalness: 0.55, emissive: 0x0a0f18, emissiveIntensity: 0.3 });
    const botSlab = new THREE.Mesh(slabGeo, botMat); botLayer.add(botSlab);
    const botEdge = makeEdges(botSlab, CYAN);
    const cores = [];
    const coreGeo = new THREE.BoxGeometry(0.26, 0.05, 0.26);
    const CG = 6, cstep = (SLAB - 0.6) / (CG - 1), c0 = -(SLAB - 0.6) / 2;
    for (let r = 0; r < CG; r++) for (let c = 0; c < CG; c++) {
        const m = new THREE.MeshStandardMaterial({ color: 0x0d1420, emissive: CYAN, emissiveIntensity: 0.12, roughness: 0.4 });
        const core = new THREE.Mesh(coreGeo, m);
        core.position.set(c0 + c * cstep, THICK / 2 + 0.03, c0 + r * cstep);
        core.userData.d = r + c; botLayer.add(core); cores.push(core);
    }

    /* vertical connectors */
    const postMat = new THREE.LineBasicMaterial({ color: 0x9fb4d6, transparent: true, opacity: 0.26 });
    const posts = [];
    const cxz = SLAB / 2 - 0.14;
    [[-cxz, -cxz], [cxz, -cxz], [-cxz, cxz], [cxz, cxz]].forEach(p => {
        const ln = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(p[0], -1, p[1]), new THREE.Vector3(p[0], 1, p[1])]), postMat);
        stack.add(ln); posts.push(ln);
    });

    /* flow particles */
    const flow = new THREE.Group(); stack.add(flow);
    const pGeo = new THREE.SphereGeometry(0.055, 10, 10);
    const particles = [];
    for (let i = 0; i < 18; i++) {
        const p = new THREE.Mesh(pGeo, new THREE.MeshBasicMaterial({ color: GOLD }));
        p.userData = { x: (Math.random() - 0.5) * (SLAB - 0.5), z: (Math.random() - 0.5) * (SLAB - 0.5), t: Math.random(), sp: 0.12 + Math.random() * 0.2 };
        flow.add(p); particles.push(p);
    }

    stack.position.y = 0.12;

    /* branch mapping (top→auto, mid→memory, bot→core) */
    const LAYERS = {
        auto:   { layer: topLayer, slab: topSlab, edge: topEdge, mat: topMat, baseEm: 0.4, sign: 1 },
        memory: { layer: midLayer, slab: midSlab, edge: midEdge, mat: midMat, baseEm: 0.0, sign: 0 },
        core:   { layer: botLayer, slab: botSlab, edge: botEdge, mat: botMat, baseEm: 0.3, sign: -1 }
    };
    const slabToKey = new Map([[topSlab, 'auto'], [midSlab, 'memory'], [botSlab, 'core']]);
    const raycaster = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    let hoverKey = null, forcedKey = null;

    function hitTest(clientX, clientY) {
        const rect = canvas.getBoundingClientRect();
        if (!rect.width || !rect.height) return null;
        ndc.x = ((clientX - rect.left) / rect.width) * 2 - 1;
        ndc.y = -((clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(ndc, camera);
        const hits = raycaster.intersectObjects([topSlab, midSlab, botSlab], false);
        return hits.length ? slabToKey.get(hits[0].object) : null;
    }
    function setHover(k) {
        if (k === hoverKey) return;
        hoverKey = k;
        canvas.style.cursor = k ? 'pointer' : 'grab';
        if (window.__onStackHover) window.__onStackHover(k);
        if (reduce) renderOnce();
    }
    // external API — left branch rows drive this
    window.__stackSetForced = function (k) { forcedKey = k; if (reduce) renderOnce(); };

    /* interaction */
    let dragging = false, lastX = 0, lastY = 0, moved = 0, autoSpin = true;
    let rotY = -0.5, tiltX = 0.0;
    canvas.addEventListener('pointerdown', e => {
        dragging = true; moved = 0; lastX = e.clientX; lastY = e.clientY; autoSpin = false;
        setHover(hitTest(e.clientX, e.clientY));
        canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', e => {
        if (dragging) {
            const dx = e.clientX - lastX, dy = e.clientY - lastY;
            moved += Math.abs(dx) + Math.abs(dy);
            rotY += dx * 0.008;
            tiltX = Math.max(-0.5, Math.min(0.6, tiltX + dy * 0.005));
            lastX = e.clientX; lastY = e.clientY;
            return;
        }
        setHover(hitTest(e.clientX, e.clientY));
    });
    canvas.addEventListener('pointerup', e => {
        dragging = false;
        const k = hitTest(e.clientX, e.clientY);
        if (moved < 6 && k) { const t = document.getElementById(SCROLL[k]); if (t) t.scrollIntoView({ behavior: 'smooth', block: 'start' }); return; }
        setTimeout(() => { autoSpin = true; }, 2600);
    });
    canvas.addEventListener('pointerleave', () => setHover(null));

    function resize() {
        const w = canvas.clientWidth, h = canvas.clientHeight;
        if (!w || !h) return;
        const pr = renderer.getPixelRatio();
        if (canvas.width !== Math.round(w * pr) || canvas.height !== Math.round(h * pr)) {
            renderer.setSize(w, h, false);
            const aspect = w / h;
            camera.aspect = aspect;
            // frame the stack for any canvas shape (short/wide or tall)
            const fov = camera.fov * Math.PI / 180;
            let dist = 1.9 / Math.sin(fov / 2);
            if (aspect < 1) dist /= aspect;         // portrait — pull back
            camera.position.set(0, dist * 0.36, dist * 0.93);
            camera.lookAt(0, -0.05, 0);
            camera.updateProjectionMatrix();
        }
    }

    const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let running = false, raf = null, T = 0;

    function applyHighlight(G) {
        const eff = forcedKey || hoverKey;
        ['auto', 'memory', 'core'].forEach(k => {
            const L = LAYERS[k];
            const on = (k === eff);
            L.edge.material.opacity = on ? 1 : 0.85;
            L.slab.scale.setScalar(on ? 1.03 : 1);
            if (k !== 'memory') L.mat.emissiveIntensity = L.baseEm + (on ? 0.5 : 0);
            const extra = on ? 0.18 : 0;
            L.layer.position.y = L.sign * G + L.sign * extra;
        });
    }

    function renderOnce() { resize(); applyHighlight(0.66); renderer.render(scene, camera); }

    function frame() {
        if (!running) return;
        resize();
        T += 0.016;
        if (autoSpin && !dragging) rotY += 0.0035;
        stack.rotation.y = rotY; stack.rotation.x = tiltX;

        const G = 0.66 + Math.sin(T * 0.7) * 0.1;
        applyHighlight(G);
        posts.forEach(ln => { ln.scale.y = (G + 0.3); });

        const wave = (T * 3) % (CG * 2 + 4);
        cores.forEach(cr => { cr.material.emissiveIntensity = 0.1 + Math.max(0, 1.3 - Math.abs(cr.userData.d - wave) * 0.7); });
        nodes.forEach(n => { n.material.emissiveIntensity = 0.5 + (Math.sin(T * 3 + n.userData.ph) + 1) * 0.35; });

        const topY = G + 0.25, botY = -G - 0.25, span = topY - botY;
        particles.forEach(p => {
            p.userData.t += p.userData.sp * 0.016;
            if (p.userData.t > 1) { p.userData.t = 0; p.userData.x = (Math.random() - 0.5) * (SLAB - 0.5); p.userData.z = (Math.random() - 0.5) * (SLAB - 0.5); }
            p.position.set(p.userData.x, topY - span * p.userData.t, p.userData.z);
            flowColor(p.userData.t, p.material.color);
        });

        renderer.render(scene, camera);
        raf = requestAnimationFrame(frame);
    }

    function start() {
        if (running) return;
        if (reduce) { renderOnce(); return; }
        running = true; raf = requestAnimationFrame(frame);
    }
    function stop() { running = false; if (raf) cancelAnimationFrame(raf); }

    const io = new IntersectionObserver(es => es.forEach(e => { e.isIntersecting ? start() : stop(); }), { threshold: 0.05 });
    io.observe(canvas);
    document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); else start(); });
})();
