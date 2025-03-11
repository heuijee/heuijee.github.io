# Markdown Blog System

이 디렉토리는 Markdown 형식으로 작성된 블로그 포스트를 포함하고 있습니다. 이 시스템은 자동으로 Markdown 파일을 읽어 블로그 페이지에 표시합니다.

## 새 블로그 포스트 추가하기

1. 이 디렉토리에 새 `.md` 파일을 설명적인 이름으로 생성하세요 (예: `my-new-blog-post.md`).
2. 파일 상단에 필수 YAML frontmatter를 포함하세요:

```md
---
title: 블로그 포스트 제목
date: YYYY-MM-DD
author: 작성자 이름
excerpt: 목록 보기에 표시될 짧은 요약
---

# 여기서부터 블로그 포스트 내용 시작

Markdown 형식으로 블로그 내용 작성...
```

3. Markdown 문법을 사용하여 블로그 내용을 작성하세요.
4. `generate-blog-content.js` 스크립트를 실행하여 블로그 콘텐츠를 자동 생성하세요:

```
node blog_posts/generate-blog-content.js
```

이 스크립트는 모든 Markdown 파일을 분석하여 `blog.js` 파일을 생성합니다.

## 자동 생성 절차

이 시스템은 다음과 같이 작동합니다:

1. 모든 Markdown 파일 (`*.md`)이 `blog_posts` 디렉토리에서 읽혀집니다.
2. 각 파일의 frontmatter와 내용이 추출됩니다.
3. 이 정보는 `blog.js` 파일을 생성하는 데 사용됩니다.
4. 생성된 `blog.js` 파일은 모든 블로그 컨텐츠를 포함하고 있어 로컬에서도 작동합니다.

## Markdown 기능

이 블로그 시스템은 다음을 지원합니다:

- 제목 (# H1, ## H2, 등)
- 목록 (순서가 있는/없는 목록)
- 링크와 이미지
- 구문 강조가 있는 코드 블록
- 표
- 인용구
- 굵은 글씨와 기울임꼴

## 코드 블록 예시

```javascript
function helloWorld() {
  console.log("안녕하세요, 세상!");
}
```

## 이미지

이미지를 `blog_posts/images` 디렉토리에 넣고 다음과 같이 참조하세요:

```md
![이미지 대체 텍스트](images/your-image.jpg)
```

## 문제 해결

블로그 포스트가 웹사이트에 나타나지 않는 경우:

1. frontmatter가 올바른지 확인하세요 (title과 date는 필수입니다)
2. Markdown 문법이 올바른지 확인하세요
3. `generate-blog-content.js` 스크립트를 다시 실행해보세요

## 작동 방식

이 시스템은 `fetch()` API의 한계를 우회하기 위해 모든 블로그 컨텐츠를 `blog.js` 파일 내에 직접 포함시킵니다. 이렇게 하면 로컬 환경에서도 블로그가 제대로 작동합니다.

파일을 수정하거나 추가한 후에는 항상 `generate-blog-content.js` 스크립트를 실행하여 `blog.js` 파일을 업데이트하세요. 