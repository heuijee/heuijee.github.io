/**
 * Blog Content Generator
 * 
 * This script:
 * 1. Reads all Markdown files in the blog_posts directory
 * 2. Generates an updated version of blog.js with all blog content embedded
 * 3. Allows working with local files without server requirements
 * 
 * Usage:
 * node generate-blog-content.js
 */

const fs = require('fs');
const path = require('path');

// Configuration
const POSTS_DIR = path.join(__dirname);
const BLOG_JS_TEMPLATE = path.join(__dirname, '..', 'js', 'blog-template.js');
const BLOG_JS_OUTPUT = path.join(__dirname, '..', 'js', 'blog.js');
const VALID_EXTENSIONS = ['.md', '.markdown'];

/**
 * Main function to generate the blog.js file
 */
function generateBlogJs() {
  console.log('Generating blog.js with embedded content...');
  
  try {
    // Get all markdown files
    const files = getMarkdownFiles(POSTS_DIR);
    
    // Skip non-post files like README.md and the index generator
    const postFiles = files.filter(file => {
      const basename = path.basename(file);
      return basename !== 'README.md' && !basename.startsWith('generate-');
    });
    
    console.log(`Found ${postFiles.length} blog posts`);
    
    // Extract metadata and content from each file
    const blogData = [];
    const blogContents = {};
    
    postFiles.forEach(file => {
      const filePath = path.join(POSTS_DIR, file);
      const fileContent = fs.readFileSync(filePath, 'utf8');
      
      // Extract frontmatter and content
      const { metadata, content } = extractFrontmatter(fileContent);
      
      // Skip files without proper frontmatter
      if (!metadata.title || !metadata.date) {
        console.warn(`Warning: ${file} is missing required frontmatter (title or date)`);
        return;
      }
      
      const id = path.basename(file, path.extname(file));
      
      // Add to blog data
      blogData.push({
        id,
        path: `blog_posts/${file}`,
        title: metadata.title,
        date: metadata.date,
        author: metadata.author || 'Heuijee Yun',
        excerpt: metadata.excerpt || ''
      });
      
      // Fix image paths in markdown content
      const fixedContent = fixImagePaths(content.trim(), id);
      
      // Add to blog contents - make sure to include the content
      blogContents[id] = fixedContent;
      
      // Debug log
      console.log(`Processed: ${id}, Content length: ${fixedContent.length} characters`);
    });
    
    // Sort posts by date (newest first)
    blogData.sort((a, b) => new Date(b.date) - new Date(a.date));
    
    // Generate blog.js content
    if (!fs.existsSync(BLOG_JS_TEMPLATE)) {
      // If template doesn't exist, create one from scratch
      generateBlogJsContent(blogData, blogContents);
    } else {
      // Use the template
      const templateContent = fs.readFileSync(BLOG_JS_TEMPLATE, 'utf8');
      const updatedContent = updateBlogJsFromTemplate(templateContent, blogData, blogContents);
      fs.writeFileSync(BLOG_JS_OUTPUT, updatedContent);
    }
    
    console.log(`Successfully generated blog.js with ${blogData.length} posts`);
    console.log(`Output file: ${BLOG_JS_OUTPUT}`);
    
    // Debug check: Log the first few characters of each blog post content
    console.log("\nContent Preview:");
    Object.keys(blogContents).forEach(key => {
      const preview = blogContents[key].substring(0, 50).replace(/\n/g, ' ');
      console.log(`${key}: ${preview}...`);
    });
    
    return true;
  } catch (error) {
    console.error('Error generating blog.js:', error);
    return false;
  }
}

/**
 * Fix image paths in markdown to ensure they load correctly on the web
 */
function fixImagePaths(content, postId) {
  // Replace relative image paths with blog_posts relative paths
  // Pattern: ![alt text](images/filename.ext) -> ![alt text](blog_posts/images/filename.ext)
  const updatedContent = content.replace(
    /!\[(.*?)\]\((images\/[^)]+)\)/g, 
    '![$1](blog_posts/$2)'
  );
  
  // Also handle HTML image tags
  const htmlImgFixed = updatedContent.replace(
    /<img\s+src="(images\/[^"]+)"/g,
    '<img src="blog_posts/$1"'
  );
  
  return htmlImgFixed;
}

/**
 * Extract frontmatter and content from markdown
 */
function extractFrontmatter(markdown) {
  // 정규식을 개선하여 frontmatter와 content를 더 정확하게 추출
  const frontmatterRegex = /^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$/;
  const match = markdown.match(frontmatterRegex);
  
  if (!match) {
    console.warn('No frontmatter found, using entire content');
    return { 
      metadata: {}, 
      content: markdown 
    };
  }
  
  const frontmatterStr = match[1];
  const content = match[2];
  
  // Parse frontmatter
  const metadata = {};
  frontmatterStr.split('\n').forEach(line => {
    const colonIndex = line.indexOf(':');
    if (colonIndex !== -1) {
      const key = line.slice(0, colonIndex).trim();
      const value = line.slice(colonIndex + 1).trim();
      metadata[key] = value;
    }
  });
  
  return { metadata, content };
}

/**
 * Get all markdown files in the specified directory
 */
function getMarkdownFiles(dir) {
  return fs.readdirSync(dir)
    .filter(file => {
      const filePath = path.join(dir, file);
      return fs.statSync(filePath).isFile() && 
             VALID_EXTENSIONS.includes(path.extname(file).toLowerCase());
    });
}

/**
 * Generate blog.js content from scratch
 */
function generateBlogJsContent(blogData, blogContents) {
  // Create blog data JSON
  const blogDataJson = JSON.stringify(blogData, null, 2)
    .replace(/^/gm, '    ')
    .replace(/^    /, '');
  
  // Create blog contents object
  let blogContentsStr = 'const blogContents = {\n';
  Object.keys(blogContents).forEach(id => {
    // Escape backticks and special chars in content
    const escapedContent = blogContents[id]
      .replace(/`/g, '\\`')
      .replace(/\$/g, '\\$');
    
    blogContentsStr += `    '${id}': \`${escapedContent}\`,\n\n`;
  });
  blogContentsStr += '};';
  
  const jsContent = `/**
 * Blog system for loading and displaying Markdown blog posts
 * Auto-generated from blog post markdown files
 * Generated on: ${new Date().toISOString()}
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

    // Blog post metadata store
    let blogPosts = ${blogDataJson};

    // Blog post content store - pre-loaded from markdown files
    ${blogContentsStr}

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
            blogListEl.innerHTML = '<div class="error-message"><p>블로그 포스트를 찾을 수 없습니다</p><p>blog_posts 디렉토리에 Markdown 파일을 추가하고 생성 스크립트를 실행하세요.</p></div>';
            return;
        }
        
        // Sort posts by date (newest first)
        posts.sort((a, b) => new Date(b.date) - new Date(a.date));
        
        // Render each post
        posts.forEach(post => {
            const formattedDate = formatDate(post.date);
            
            const postEl = document.createElement('a');
            postEl.href = \`?post=\${post.id}\`;
            postEl.classList.add('blog-link');
            postEl.dataset.postId = post.id;
            postEl.innerHTML = \`
                <div class="blog-item">
                    <p class="blog-date">\${formattedDate}</p>
                    <h3 class="blog-title">\${post.title}</h3>
                    <p class="blog-excerpt">\${post.excerpt}</p>
                </div>
            \`;
            
            // Add click event to load the blog post
            postEl.addEventListener('click', function(e) {
                e.preventDefault();
                loadBlogPost(post.id);
                // Update URL without page reload
                window.history.pushState({}, '', \`?post=\${post.id}\`);
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
            showError('블로그 포스트를 찾을 수 없습니다');
            return;
        }
        
        // Update post metadata in the UI
        blogPostTitleEl.textContent = post.title;
        blogPostDateEl.textContent = formatDate(post.date);
        blogPostAuthorEl.textContent = post.author;
        
        // Debug content presence
        console.log("Blog post ID:", postId);
        console.log("Available contents:", Object.keys(blogContents));
        console.log("Content exists:", !!blogContents[postId]);
        
        // Check if we have the content in our local store
        if (blogContents[postId]) {
            // Use the pre-loaded content
            const content = blogContents[postId];
            
            // Convert markdown to HTML and render
            blogPostContentEl.innerHTML = marked.parse(content);
            
            // Fix any remaining image paths in generated HTML
            document.querySelectorAll('#blog-post-content img').forEach(img => {
                const src = img.getAttribute('src');
                if (src && src.startsWith('images/')) {
                    img.src = 'blog_posts/' + src;
                }
            });
            
            // Apply syntax highlighting to code blocks
            document.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightBlock(block);
            });
            
            // Show the blog post view
            showBlogPost();
        } else {
            showError(\`"\${post.title}" 블로그 내용을 찾을 수 없습니다. 생성 스크립트를 실행해주세요.\`);
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
        blogPostContentEl.innerHTML = \`
            <div class="error-message">
                <p>\${message}</p>
                <p>블로그 포스트를 읽을 수 없습니다. 관리자에게 문의하세요.</p>
            </div>
        \`;
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
});`;

  fs.writeFileSync(BLOG_JS_OUTPUT, jsContent);
}

/**
 * Update blog.js from a template file
 */
function updateBlogJsFromTemplate(templateContent, blogData, blogContents) {
  // Create blog data JSON
  const blogDataJson = JSON.stringify(blogData, null, 2)
    .replace(/^/gm, '    ')
    .replace(/^    /, '');
  
  // Create blog contents object
  let blogContentsStr = 'const blogContents = {\n';
  Object.keys(blogContents).forEach(id => {
    // Escape backticks and special chars in content
    const escapedContent = blogContents[id]
      .replace(/`/g, '\\`')
      .replace(/\$/g, '\\$');
    
    blogContentsStr += `    '${id}': \`${escapedContent}\`,\n\n`;
  });
  blogContentsStr += '};';
  
  // Debug - check content before substitution
  console.log("\nIDs in blogContents object:", Object.keys(blogContents));
  
  // Replace placeholders in the template
  const result = templateContent
    .replace('// AUTO-GENERATED BLOG DATA', `let blogPosts = ${blogDataJson};`)
    .replace('// AUTO-GENERATED BLOG CONTENTS', blogContentsStr);
  
  return result;
}

// Execute the generator
generateBlogJs(); 