// 페이지가 로드되면 실행
document.addEventListener('DOMContentLoaded', function() {
    // 헤더 스타일 변경을 위한 스크롤 이벤트 리스너
    window.addEventListener('scroll', function() {
        var header = document.querySelector('.site-header');
        if (header && window.scrollY > 50) {
            header.classList.add('scrolled');
        } else if (header) {
            header.classList.remove('scrolled');
        }
    });

    // 홈페이지에서만 실행되는 코드
    if (document.body.classList.contains('homepage')) {
        // 소셜 아이콘 애니메이션
        const socialIcons = document.querySelectorAll('.social-icon');
        socialIcons.forEach((icon, index) => {
            setTimeout(() => {
                icon.classList.add('visible');
            }, 200 * index);
        });
    }

    // 페이지 로드 시 요소 페이드인 효과
    const fadeElements = document.querySelectorAll('.section-title, .publications-list li, .highlight-card, .project-card, .cv-item');
    
    if (fadeElements.length > 0) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.2 });
        
        fadeElements.forEach(el => {
            observer.observe(el);
        });
    }

    // 컨택트 폼 제출 처리 (실제 제출 기능은 구현되지 않음)
    const contactForm = document.getElementById('contact-form');
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            alert('메시지가 전송되었습니다. (데모용 알림)');
            contactForm.reset();
        });
    }
}); 