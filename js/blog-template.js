/**
 * Blog system for loading and displaying Markdown blog posts
 * Template file for auto-generation
 */
document.addEventListener('DOMContentLoaded', function() {
    // Configure marked.js settings
    marked.setOptions({
        highlight: function(code, lang) {
            if (lang && hljs.getLanguage(lang)) {
                return hljs.highlight(code, { language: lang }).value;
            }
            return hljs.highlightAuto(code).value;
        },
        breaks: true,
        gfm: true, // GitHub Flavored Markdown
        tables: true
    });

    // Elements
    const blogListEl = document.getElementById('blog-list');
    const blogPostEl = document.getElementById('blog-post');
    const blogPostTitleEl = document.querySelector('.blog-post-title');
    const blogPostDateEl = document.querySelector('.blog-post-date');
    const blogPostAuthorEl = document.querySelector('.blog-post-author');
    const blogPostContentEl = document.getElementById('blog-post-content');
    const backButton = document.getElementById('back-to-blogs');

    // Blog post metadata store - Will be replaced by auto-generated content
    // AUTO-GENERATED BLOG DATA

    // Blog post content store - Will be replaced by auto-generated content
    // AUTO-GENERATED BLOG CONTENTS

    // Initialize blog
    init();

    /**
     * Initialize the blog
     */
    function init() {
        // Only run on the blogs page
        if (!document.body.classList.contains('blogs-page')) return;
        
        // Render blog posts
        renderBlogList(blogPosts);
        
        // Set up back button
        backButton.addEventListener('click', showBlogList);
        
        // Check for direct post URL
        const urlParams = new URLSearchParams(window.location.search);
        const postParam = urlParams.get('post');
        
        if (postParam) {
            loadBlogPost(postParam);
        }
    }

    /**
     * Render the blog list from the metadata
     */
    function renderBlogList(posts) {
        // Clear loading indicator
        blogListEl.innerHTML = '';
        
        if (!posts || posts.length === 0) {
            blogListEl.innerHTML = '<div class="error-message"><p>No blog posts found</p><p>Please add Markdown files to the blog_posts directory and run the generator script.</p></div>';
            return;
        }
        
        // Sort posts by date (newest first)
        posts.sort((a, b) => new Date(b.date) - new Date(a.date));
        
        // Render each post
        posts.forEach(post => {
            const formattedDate = formatDate(post.date);
            
            const postEl = document.createElement('a');
            postEl.href = `?post=${post.id}`;
            postEl.classList.add('blog-link');
            postEl.dataset.postId = post.id;
            postEl.innerHTML = `
                <div class="blog-item">
                    <p class="blog-date">${formattedDate}</p>
                    <h3 class="blog-title">${post.title}</h3>
                    <p class="blog-excerpt">${post.excerpt}</p>
                </div>
            `;
            
            // Add click event to load the blog post
            postEl.addEventListener('click', function(e) {
                e.preventDefault();
                loadBlogPost(post.id);
                // Update URL without page reload
                window.history.pushState({}, '', `?post=${post.id}`);
            });
            
            blogListEl.appendChild(postEl);
        });
    }

    /**
     * Load and display a blog post by its ID
     */
    function loadBlogPost(postId) {
        // Find the post metadata
        const post = blogPosts.find(p => p.id === postId);
        
        if (!post) {
            showError('Blog post not found');
            return;
        }
        
        // Update post metadata in the UI
        blogPostTitleEl.textContent = post.title;
        blogPostDateEl.textContent = formatDate(post.date);
        blogPostAuthorEl.textContent = post.author;
        
        // Check if we have the content in our local store
        if (blogContents && blogContents[postId]) {
            // Use the pre-loaded content
            const content = blogContents[postId];
            
            // Convert markdown to HTML and render
            blogPostContentEl.innerHTML = marked.parse(content);
            
            // Apply syntax highlighting to code blocks
            document.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightBlock(block);
            });
            
            // Show the blog post view
            showBlogPost();
        } else {
            showError(`Blog content not found for "${post.title}". Please run the generator script.`);
        }
    }

    /**
     * Show the blog list view, hide the blog post view
     */
    function showBlogList() {
        blogPostEl.style.display = 'none';
        blogListEl.style.display = 'flex';
        
        // Update URL without page reload
        window.history.pushState({}, '', 'blogs.html');
    }

    /**
     * Show the blog post view, hide the blog list
     */
    function showBlogPost() {
        blogListEl.style.display = 'none';
        blogPostEl.style.display = 'block';
    }

    /**
     * Show an error message in the blog content area
     */
    function showError(message) {
        blogPostContentEl.innerHTML = `
            <div class="error-message">
                <p>${message}</p>
                <p>블로그 포스트를 읽을 수 없습니다. 관리자에게 문의하세요.</p>
            </div>
        `;
        showBlogPost();
    }

    /**
     * Format a date string as "Month DD, YYYY"
     */
    function formatDate(dateStr) {
        const date = new Date(dateStr);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
    }

    // Handle browser back/forward navigation
    window.addEventListener('popstate', function() {
        const urlParams = new URLSearchParams(window.location.search);
        const postParam = urlParams.get('post');
        
        if (postParam) {
            loadBlogPost(postParam);
        } else {
            showBlogList();
        }
    });
}); 