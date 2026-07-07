/* ==========================================================================
   Header state on scroll
   ========================================================================== */
document.addEventListener('DOMContentLoaded', function () {
    window.addEventListener('scroll', function () {
        const header = document.querySelector('.top-nav');
        if (!header) return;
        if (window.scrollY > 8) header.classList.add('scrolled');
        else header.classList.remove('scrolled');
    });
});

/* ==========================================================================
   Contact form (EmailJS) — contact page only
   ========================================================================== */
document.addEventListener('DOMContentLoaded', function () {
    const contactForm = document.getElementById('contact-form');
    if (!contactForm || typeof emailjs === 'undefined') return;
    emailjs.init("qbvzcoL06-OBJC1lD");
    contactForm.addEventListener('submit', function (event) {
        event.preventDefault();
        const name = (document.getElementById('name') || {}).value?.trim();
        const email = (document.getElementById('email') || {}).value?.trim();
        const message = (document.getElementById('message') || {}).value?.trim();
        if (!name || !email || !message) { alert("Please fill in all fields before submitting."); return; }
        emailjs.send("service_36ycsph", "template_6djy2mp", { user_name: name, user_email: email, user_message: message })
            .then(function () { alert("Your message has been sent successfully!"); contactForm.reset(); })
            .catch(function (err) { alert("Failed to send the message."); console.error("EmailJS Error:", err); });
    });
});

/* ==========================================================================
   Legacy research accordion (only if the old markup is present)
   ========================================================================== */
document.addEventListener("DOMContentLoaded", function () {
    const researchMain = document.getElementById("research-main");
    const container = document.getElementById("research-details-container");
    const backButton = document.getElementById("back-button");
    if (!researchMain || !container || !backButton) return;
    const items = document.querySelectorAll(".research-item");
    const details = document.querySelectorAll(".research-details");
    container.style.display = "none";
    items.forEach(item => item.addEventListener("click", function () {
        const sel = document.getElementById(this.getAttribute("data-research"));
        if (!sel) return;
        researchMain.style.opacity = "0"; researchMain.style.transform = "translateY(-10px)";
        setTimeout(() => { researchMain.style.display = "none"; }, 300);
        details.forEach(d => { d.style.display = "none"; d.classList.remove("fade-in"); });
        container.style.display = "block"; sel.style.display = "block";
        setTimeout(() => { container.classList.add("fade-in"); sel.classList.add("fade-in"); }, 10);
    }));
    backButton.addEventListener("click", function () {
        container.classList.remove("fade-in");
        details.forEach(d => d.classList.remove("fade-in"));
        setTimeout(() => {
            container.style.display = "none"; researchMain.style.display = "block";
            setTimeout(() => { researchMain.style.opacity = "1"; researchMain.style.transform = "translateY(0)"; }, 10);
        }, 300);
    });
});

/* ==========================================================================
   Scroll reveal — fade/rise content as it enters the viewport
   ========================================================================== */
(function scrollReveal() {
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    if (!('IntersectionObserver' in window)) return;

    document.querySelectorAll('.pub, .project-card, .interest, details.topic, .gadget, .timeline-item, .blog-item, .branch-head, .why-card, .ev-row')
        .forEach(el => el.classList.add('reveal'));

    const els = document.querySelectorAll('.reveal');
    if (!els.length) return;
    const io = new IntersectionObserver((entries) => {
        entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
    }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
    els.forEach(el => io.observe(el));
})();

/* ==========================================================================
   Homepage — sticky stack-rail highlight + number count-up
   ========================================================================== */
(function homepageExtras() {
    if (!document.body.classList.contains('home')) return;

    // rail highlight synced to the layer near the viewport centre
    const rail = document.querySelectorAll('.stack-rail li[data-rail]');
    const layers = document.querySelectorAll('.layer[id]');
    if (rail.length && layers.length && 'IntersectionObserver' in window) {
        const setActive = (id) => rail.forEach(li => li.classList.toggle('active', li.dataset.rail === id));
        const io = new IntersectionObserver((entries) => {
            entries.forEach(e => { if (e.isIntersecting) setActive(e.target.id); });
        }, { rootMargin: '-45% 0px -45% 0px', threshold: 0 });
        layers.forEach(l => io.observe(l));
        setActive(layers[0].id);
    }

    // count-up numbers when they scroll into view
    const nums = document.querySelectorAll('.stat-n[data-count]');
    if (nums.length && 'IntersectionObserver' in window &&
        !(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches)) {
        const run = (el) => {
            const target = parseInt(el.dataset.count, 10) || 0;
            const dur = 1100, t0 = performance.now();
            const tick = (now) => {
                const p = Math.min((now - t0) / dur, 1);
                const eased = 1 - Math.pow(1 - p, 3); // ease-out cubic
                el.textContent = Math.round(target * eased);
                if (p < 1) requestAnimationFrame(tick);
            };
            requestAnimationFrame(tick);
        };
        const io = new IntersectionObserver((entries) => {
            entries.forEach(e => { if (e.isIntersecting) { run(e.target); io.unobserve(e.target); } });
        }, { threshold: 0.5 });
        nums.forEach(n => io.observe(n));
    }
})();
