// 페이지가 로드되면 실행
document.addEventListener('DOMContentLoaded', function () {
    // 헤더 스크롤 상태
    window.addEventListener('scroll', function () {
        var header = document.querySelector('.site-header');
        if (header && window.scrollY > 50) header.classList.add('scrolled');
        else if (header) header.classList.remove('scrolled');
    });

    // 홈페이지 소셜 아이콘 순차 등장
    if (document.body.classList.contains('homepage')) {
        document.querySelectorAll('.social-icon').forEach((icon, i) => {
            setTimeout(() => icon.classList.add('visible'), 120 * i);
        });
    }
});

/* ==========================================================================
   Contact form (EmailJS) — contact 페이지에서만
   ========================================================================== */
document.addEventListener('DOMContentLoaded', function () {
    const contactForm = document.getElementById('contact-form');
    if (!contactForm || typeof emailjs === 'undefined') return;

    emailjs.init("qbvzcoL06-OBJC1lD");

    contactForm.addEventListener('submit', function (event) {
        event.preventDefault();
        const nameField = document.getElementById('name');
        const emailField = document.getElementById('email');
        const messageField = document.getElementById('message');
        if (!nameField || !emailField || !messageField) return;

        const name = nameField.value.trim();
        const email = emailField.value.trim();
        const message = messageField.value.trim();
        if (!name || !email || !message) { alert("Please fill in all fields before submitting."); return; }

        emailjs.send("service_36ycsph", "template_6djy2mp", {
            user_name: name, user_email: email, user_message: message
        }).then(function () {
            alert("Your message has been sent successfully!");
            contactForm.reset();
        }).catch(function (error) {
            alert("Failed to send the message. Please check the console for details.");
            console.error("EmailJS Error: ", error);
        });
    });
});

/* ==========================================================================
   Legacy research accordion (only fires on the old markup; new page uses
   native <details> so this returns early)
   ========================================================================== */
document.addEventListener("DOMContentLoaded", function () {
    const researchItems = document.querySelectorAll(".research-item");
    const researchMain = document.getElementById("research-main");
    const researchDetailsContainer = document.getElementById("research-details-container");
    const researchDetails = document.querySelectorAll(".research-details");
    const backButton = document.getElementById("back-button");
    if (!researchMain || !researchDetailsContainer || !backButton) return;

    researchDetailsContainer.style.display = "none";
    researchItems.forEach(item => {
        item.addEventListener("click", function () {
            const selected = document.getElementById(this.getAttribute("data-research"));
            if (!selected) return;
            researchMain.style.opacity = "0";
            researchMain.style.transform = "translateY(-10px)";
            setTimeout(() => { researchMain.style.display = "none"; }, 300);
            researchDetails.forEach(d => { d.style.display = "none"; d.classList.remove("fade-in"); });
            researchDetailsContainer.style.display = "block";
            selected.style.display = "block";
            setTimeout(() => { researchDetailsContainer.classList.add("fade-in"); selected.classList.add("fade-in"); }, 10);
        });
    });
    backButton.addEventListener("click", function () {
        researchDetailsContainer.classList.remove("fade-in");
        researchDetails.forEach(d => d.classList.remove("fade-in"));
        setTimeout(() => {
            researchDetailsContainer.style.display = "none";
            researchMain.style.display = "block";
            setTimeout(() => { researchMain.style.opacity = "1"; researchMain.style.transform = "translateY(0)"; }, 10);
        }, 300);
    });
});

/* ==========================================================================
   Signature backdrop — "ALGORITHM → SILICON" descent.
   Tokens fall from software (top) to silicon (bottom), morphing at each
   abstraction layer: neuron/spike → RTL glyph → logic gate → std cell → metal.
   Runs on every page; mobile-aware; respects prefers-reduced-motion.
   ========================================================================== */
(function stackDescent() {
    const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const canvas = document.createElement('canvas');
    canvas.id = 'stack-bg';
    canvas.setAttribute('aria-hidden', 'true');
    Object.assign(canvas.style, {
        position: 'fixed', top: '0', left: '0', width: '100%', height: '100%',
        zIndex: '-1', pointerEvents: 'none'
    });
    (document.body || document.documentElement).appendChild(canvas);
    const ctx = canvas.getContext('2d');

    const LABELS = ['ALGORITHM', 'RTL', 'LOGIC', 'CELLS', 'SILICON'];
    const GLYPHS = ['0', '1', '{', '}', '<', '>', ';', 'λ', '&', '|'];
    let w = 0, h = 0, dpr = 1, tokens = [], scanY = 0, showLabels = false;

    function rnd(a, b) { return a + Math.random() * (b - a); }

    // color down the stack: soft cyan → dusty rose → soft gold
    function colr(f) {
        let r, g, b;
        if (f < 0.5) { const t = f / 0.5; r = Math.round(86 + (207 - 86) * t); g = Math.round(194 + (130 - 194) * t); b = Math.round(214 + (168 - 214) * t); }
        else { const t = (f - 0.5) / 0.5; r = Math.round(207 + (224 - 207) * t); g = Math.round(130 + (179 - 130) * t); b = Math.round(168 + (106 - 168) * t); }
        return r + ',' + g + ',' + b;
    }

    function build() {
        dpr = Math.min(window.devicePixelRatio || 1, 2);
        w = window.innerWidth; h = window.innerHeight;
        canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        showLabels = w >= 940;

        const n = Math.max(10, Math.min(64, Math.floor(w / 22)));
        tokens = [];
        for (let i = 0; i < n; i++) {
            tokens.push({
                x: Math.random() * w,
                y: Math.random() * h,
                sp: rnd(0.35, 1.15),
                wob: rnd(3, 12),
                ph: Math.random() * Math.PI * 2,
                glyph: GLYPHS[(Math.random() * GLYPHS.length) | 0],
                spike: Math.random() < 0.4,
                col: rnd(0, 5)
            });
        }
    }

    function draw() {
        if (!running) return;
        ctx.clearRect(0, 0, w, h);

        // layer separators + labels
        for (let i = 1; i < 5; i++) {
            const y = h * i / 5;
            ctx.strokeStyle = 'rgba(120,165,210,0.045)';
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
        }
        if (showLabels) {
            ctx.font = '700 10px "JetBrains Mono", monospace';
            for (let i = 0; i < 5; i++) {
                const y = h * (i + 0.5) / 5;
                ctx.fillStyle = 'rgba(' + colr(i / 4) + ',0.14)';
                ctx.fillText(LABELS[i], 12, y);
                ctx.fillText('L' + i, w - 30, y);
            }
        }

        // synthesis scan pass (top → bottom, slow)
        scanY = (scanY + h * 0.0012 + 0.3) % (h + 60);
        const sg = ctx.createLinearGradient(0, scanY - 34, 0, scanY + 34);
        const sc = colr(Math.min(1, scanY / h));
        sg.addColorStop(0, 'rgba(' + sc + ',0)');
        sg.addColorStop(0.5, 'rgba(' + sc + ',0.05)');
        sg.addColorStop(1, 'rgba(' + sc + ',0)');
        ctx.fillStyle = sg;
        ctx.fillRect(0, scanY - 34, w, 68);

        // tokens
        tokens.forEach(tk => {
            tk.y += tk.sp;
            tk.ph += 0.02;
            if (tk.y > h + 12) {
                tk.y = -12; tk.x = Math.random() * w; tk.sp = rnd(0.35, 1.15);
                tk.glyph = GLYPHS[(Math.random() * GLYPHS.length) | 0]; tk.spike = Math.random() < 0.4;
            }
            const f = Math.max(0, Math.min(1, tk.y / h));
            const st = Math.min(4, Math.floor(f * 5));
            const x = tk.x + Math.sin(tk.ph) * tk.wob;
            const y = tk.y;
            const col = colr(f);
            ctx.fillStyle = 'rgba(' + col + ',0.62)';
            ctx.strokeStyle = 'rgba(' + col + ',0.62)';
            ctx.shadowColor = 'rgba(' + col + ',0.8)';
            ctx.shadowBlur = 6;
            ctx.lineWidth = 1.3;

            if (st === 0) {                         // neuron / spike
                ctx.beginPath(); ctx.arc(x, y, 2.4, 0, Math.PI * 2); ctx.fill();
                if (tk.spike) {
                    ctx.shadowBlur = 0;
                    ctx.globalAlpha = 0.5;
                    ctx.beginPath(); ctx.arc(x, y, 5 + (Math.sin(tk.ph * 3) + 1) * 2, 0, Math.PI * 2); ctx.stroke();
                    ctx.globalAlpha = 1;
                }
            } else if (st === 1) {                  // RTL glyph
                ctx.shadowBlur = 4;
                ctx.font = '11px "JetBrains Mono", monospace';
                ctx.fillText(tk.glyph, x - 3, y + 4);
            } else if (st === 2) {                  // logic gate (triangle)
                ctx.beginPath();
                ctx.moveTo(x - 4, y - 4); ctx.lineTo(x - 4, y + 4); ctx.lineTo(x + 4, y);
                ctx.closePath(); ctx.stroke();
                ctx.shadowBlur = 0;
                ctx.beginPath(); ctx.arc(x + 5.5, y, 1.1, 0, Math.PI * 2); ctx.stroke();
            } else if (st === 3) {                  // standard cell (box)
                ctx.shadowBlur = 4;
                ctx.strokeRect(x - 4, y - 3, 8, 6);
            } else {                                 // metal
                ctx.shadowBlur = 5;
                ctx.fillRect(x - 6, y - 1, 12, 2);
            }
            ctx.shadowBlur = 0;
        });

        // metal bus at the very bottom
        ctx.fillStyle = 'rgba(224,179,106,0.14)';
        ctx.fillRect(0, h - 3, w, 2);

        raf = requestAnimationFrame(draw);
    }

    let raf = null, running = false;
    function start() { if (!running && !reduce) { running = true; raf = requestAnimationFrame(draw); } }
    function stop() { running = false; if (raf) cancelAnimationFrame(raf); }

    let rt = null;
    window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(build, 200); });
    document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); else start(); });

    build();
    start();   // no-op under reduced motion; body glow remains as a calm backdrop
})();

/* ==========================================================================
   Scroll reveal — content slides in as it enters the viewport
   ========================================================================== */
(function scrollReveal() {
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    if (!('IntersectionObserver' in window)) return;

    const els = document.querySelectorAll('.pub, .project-card, .interest, details.topic, .gadget, .timeline-item, .blog-item, .branch-head, .why-card, .ev-row');
    if (!els.length) return;

    els.forEach(el => el.classList.add('reveal'));
    const io = new IntersectionObserver((entries) => {
        entries.forEach(e => {
            if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
        });
    }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });
    els.forEach(el => io.observe(el));
})();
