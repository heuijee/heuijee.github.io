// 페이지가 로드되면 실행
document.addEventListener('DOMContentLoaded', function() {
    // 헤더 스타일 변경을 위한 스크롤 이벤트 리스너
    window.addEventListener('scroll', function() {
        var header = document.querySelector('header');
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
    const fadeElements = document.querySelectorAll('.page-title, .publication-item, .project-item, .cv-item, .blog-item');
    
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

    // 애니메이션 클래스 적용
    setTimeout(() => {
        const profileElements = document.querySelectorAll('.animate-profile');
        profileElements.forEach(el => {
            el.style.opacity = '1';
        });
        
        const titleElement = document.querySelector('.animate-title');
        if (titleElement) {
            titleElement.style.opacity = '1';
        }
        
        setTimeout(() => {
            const contentElement = document.querySelector('.animate-content');
            if (contentElement) {
                contentElement.style.opacity = '1';
            }
        }, 300);
    }, 100);

    // 컨택트 폼 제출 처리
    const contactForm = document.getElementById('contact-form');
    if (contactForm) {
        // 이미 HTML에 인라인 이벤트 핸들러가 있으므로 여기서는 필요하지 않음
        // handleSubmit 함수는 contact.html에 정의되어 있습니다
    }
}); 