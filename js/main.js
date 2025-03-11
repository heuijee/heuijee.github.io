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

    // 프로젝트 페이지에서만 실행
    if (document.body.classList.contains('projects-page')) {
        // 프로젝트 리스트의 스크롤 이벤트
        const projectsList = document.querySelector('.projects-list');
        if (projectsList) {
            // 초기 스크롤 위치 저장
            let lastScrollTop = 0;
            
            projectsList.addEventListener('scroll', function() {
                // 스크롤 이벤트 발생 시 현재 스크롤 위치 체크
                const st = this.scrollTop;
                
                // Page title에 스크롤 상태 클래스 추가
                const pageTitle = document.querySelector('.page-title');
                if (pageTitle) {
                    if (st > 10) {
                        pageTitle.classList.add('scrolled');
                    } else {
                        pageTitle.classList.remove('scrolled');
                    }
                }
                
                // 마지막 스크롤 위치 업데이트
                lastScrollTop = st;
            });
        }
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
}); 

document.addEventListener('DOMContentLoaded', function () {
    //  EmailJS 초기화 (올바른 Public Key 입력)
    emailjs.init("qbvzcoL06-OBJC1lD");

    //  Contact Form 찾기
    const contactForm = document.getElementById('contact-form');
    if (!contactForm) {
        console.error("Contact form not found.");
        return;
    }

    //  Contact Form 제출 이벤트 처리
    contactForm.addEventListener('submit', function (event) {
        event.preventDefault();

        //  폼 입력 요소 가져오기
        const nameField = document.getElementById('name');
        const emailField = document.getElementById('email');
        const messageField = document.getElementById('message');

        if (!nameField || !emailField || !messageField) {
            console.error("Form fields are missing.");
            return;
        }

        //  입력값 가져오기 (trim()으로 공백 제거)
        const name = nameField.value.trim();
        const email = emailField.value.trim();
        const message = messageField.value.trim();

        //  디버깅 로그 추가
        console.log("Submitting Form Data:", { name, email, message });

        //  빈 값이 있는지 확인
        if (!name || !email || !message) {
            alert("Please fill in all fields before submitting.");
            return;
        }

        //  EmailJS로 데이터 전송
        emailjs.send("service_36ycsph", "template_6djy2mp", {
            user_name: name,   //  EmailJS 템플릿과 일치하는 변수명
            user_email: email, //  EmailJS 템플릿과 일치하는 변수명
            user_message: message //  EmailJS 템플릿과 일치하는 변수명
        })
        .then(function(response) {
            alert("Your message has been sent successfully!");
            contactForm.reset(); //  폼 초기화
        })
        .catch(function(error) {
            alert("Failed to send the message. Please check the console for details.");
            console.error("EmailJS Error: ", error);
        });
    });
});


document.addEventListener("DOMContentLoaded", function () {
    const researchItems = document.querySelectorAll(".research-item");
    const researchMain = document.getElementById("research-main");
    const researchDetailsContainer = document.getElementById("research-details-container");
    const researchDetails = document.querySelectorAll(".research-details");
    const backButton = document.getElementById("back-button");

    // 연구 상세 내용을 기본적으로 숨김
    researchDetailsContainer.style.display = "none";

    researchItems.forEach(item => {
        item.addEventListener("click", function () {
            const researchId = this.getAttribute("data-research");
            const selectedResearch = document.getElementById(researchId);

            if (!selectedResearch) {
                console.error("Research ID not found:", researchId);
                return;
            }

            // 기존 Research Interests, Current Research, Research Topics 숨기기
            researchMain.style.opacity = "0";
            researchMain.style.transform = "translateY(-10px)";
            setTimeout(() => {
                researchMain.style.display = "none";
            }, 300);

            // 모든 연구 상세 내용 숨기기
            researchDetails.forEach(detail => {
                detail.style.display = "none";
                detail.classList.remove("fade-in");
            });

            // 선택한 연구 내용 표시 (애니메이션 적용)
            researchDetailsContainer.style.display = "block";
            selectedResearch.style.display = "block";

            // 애니메이션 적용 (서서히 나타나도록 설정)
            setTimeout(() => {
                researchDetailsContainer.classList.add("fade-in");
                selectedResearch.classList.add("fade-in");
            }, 10);
        });
    });

    // 뒤로 가기 버튼 클릭 시 기존 목록 다시 보이기
    backButton.addEventListener("click", function () {
        researchDetailsContainer.classList.remove("fade-in");
        researchDetails.forEach(detail => {
            detail.classList.remove("fade-in");
        });

        setTimeout(() => {
            researchDetailsContainer.style.display = "none";
            researchMain.style.display = "block";
            setTimeout(() => {
                researchMain.style.opacity = "1";
                researchMain.style.transform = "translateY(0)";
            }, 10);
        }, 300);
    });
});
