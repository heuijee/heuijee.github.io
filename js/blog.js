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
    let blogPosts = [
      {
        "id": "03_Optimization of the Hypervisor",
        "path": "blog_posts/03_Optimization of the Hypervisor.md",
        "title": "Optimization of the Hypervisor",
        "date": "2025-03-11",
        "author": "Heuijee Yun",
        "excerpt": "Methods for optimizing hypervisor performance by addressing memory management, I/O efficiency, and CPU scheduling to enhance virtualization efficiency."
      },
      {
        "id": "02_Introduction to Neural Network Architectures",
        "path": "blog_posts/02_Introduction to Neural Network Architectures.md",
        "title": "Introduction to Neural Network Architectures",
        "date": "2025-02-15",
        "author": "Heuijee Yun",
        "excerpt": "Overview of modern neural network architectures including CNNs, RNNs, Transformers, and recent advancements like Vision Transformers (ViT) and Graph Neural Networks (GNNs)."
      },
      {
        "id": "01_The Evolution of Computer Architecture",
        "path": "blog_posts/01_The Evolution of Computer Architecture.md",
        "title": "The Evolution of Computer Architecture",
        "date": "2024-12-10",
        "author": "Heuijee Yun",
        "excerpt": "The evolution of computer architecture such as Von Neumann architecture, RISC vs. CISC, parallel computing, emerging technologies like quantum and neuromorphic computing, and future trends in processor design."
      }
    ];

    // Blog post content store - Will be replaced by auto-generated content
    const blogContents = {
    '01_The Evolution of Computer Architecture': `Computer architecture has evolved significantly since the early days of computing. This post explores key milestones and emerging trends in processor design.

## Classical Von Neumann Architecture

The classical von Neumann architecture, introduced in 1945, consists of:

1. Central Processing Unit (CPU)
2. Memory Unit
3. Input/Output Systems
4. Control Unit

This architecture still influences modern designs despite its memory bottleneck (the "von Neumann bottleneck").

## RISC vs. CISC

The debate between Reduced Instruction Set Computing (RISC) and Complex Instruction Set Computing (CISC) has shaped processor development:

| Characteristic         | RISC                 | CISC                     |
| ---------------------- | -------------------- | ------------------------ |
| Instruction complexity | Simple, fixed-length | Complex, variable-length |
| Addressing modes       | Few                  | Many                     |
| Execution              | Mostly hardwired     | Often microprogrammed    |
| Registers              | Many                 | Fewer                    |
| Examples               | ARM, MIPS, RISC-V    | x86, x86-64              |

## Parallel Computing Paradigms

### Multi-core Processing

Multi-core processors have become standard:

\`\`\`c
// Simple example of parallel programming with OpenMP
#include <omp.h>
#include <stdio.h>

int main() {
    #pragma omp parallel
    {
        int thread_id = omp.get_thread_num();
        printf("Hello from thread %d\n", thread_id);
    }
    return 0;
}
\`\`\`

### GPU Computing

Graphics Processing Units (GPUs) excel at parallel tasks:

\`\`\`cuda
// Simple CUDA kernel
__global__ void vector_add(float *A, float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
\`\`\`

## Emerging Architectures

### Quantum Computing

Quantum computers use qubits and quantum principles to tackle specific problems exponentially faster than classical computers.

### Neuromorphic Computing

Neuromorphic systems mimic the structure and function of the human brain:

- Parallel processing
- Event-driven computation
- Low power consumption
- Integrated memory and processing

## Future Directions

The end of Moore's Law is driving innovation in:

- Domain-specific architectures
- Near-memory processing
- Approximate computing
- 3D integration

## Conclusion

Computer architecture continues to evolve beyond the constraints of traditional designs toward specialized, heterogeneous, and energy-efficient systems.

## References

1. Hennessy, J. L., & Patterson, D. A. (2022). Computer Architecture: A Quantitative Approach (7th ed.)
2. Asanović, K., et al. (2023). "The New Landscape of Computer Architecture"`,

    '02_Introduction to Neural Network Architectures': `Neural networks have revolutionized machine learning across numerous domains. This post explores key architectures that have driven recent advances.

## Convolutional Neural Networks (CNNs)

CNNs have transformed computer vision through their ability to automatically learn hierarchical features from images:

\`\`\`python
import tensorflow as tf

def create_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model
\`\`\`

## Recurrent Neural Networks (RNNs)

RNNs and their variants (LSTM, GRU) excel at sequence modeling tasks like natural language processing:

![RNN Structure](blog_posts/images/rnn_diagram.jpg)

Key components of an LSTM cell include:
- Input gate
- Forget gate
- Output gate
- Cell state

## Transformer Architecture

Transformers have become the dominant paradigm for NLP tasks:

\`\`\`python
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
\`\`\`

## Recent Advances

### Vision Transformers (ViT)

Vision Transformers have challenged CNNs for image recognition tasks by applying transformer-based attention to image patches.

### Graph Neural Networks (GNNs)

GNNs process data represented as graphs, enabling applications in:
- Social network analysis
- Molecular property prediction
- Recommendation systems

## Conclusion

The evolution of neural network architectures continues to drive progress in artificial intelligence, enabling increasingly sophisticated applications across domains.

## References

1. LeCun, Y., et al. (2024). "Deep Learning: Past, Present, and Future"
2. Vaswani, A., et al. (2017). "Attention Is All You Need"`,

    '03_Optimization of the Hypervisor': `Hypervisors are a critical component in cloud computing and virtualization technologies. In this post, I'll discuss some techniques for optimizing hypervisor performance.

## Background

A hypervisor, also known as a virtual machine monitor (VMM), is software that creates and runs virtual machines. It allows multiple operating systems to share a single hardware host.

## Common Performance Issues

When working with hypervisors, you might encounter these performance bottlenecks:

- Memory overhead
- I/O latency
- CPU scheduling inefficiencies

## Optimization Techniques

### 1. Memory Management

Implementing page sharing and ballooning can significantly reduce memory overhead:

\`\`\`c
void optimize_memory() {
    // Implementation details
    enable_page_sharing();
    configure_memory_ballooning();
}
\`\`\`

### 2. I/O Optimization

Direct device assignment can bypass the virtualization layer:

- Pass-through PCIe devices
- Use SR-IOV for network cards
- Implement virtio drivers

### 3. CPU Scheduling

Careful CPU pinning and NUMA awareness improves performance:

\`\`\`c
// Pin vCPUs to physical cores
for (int i = 0; i < num_vcpus; i++) {
    pin_vcpu_to_physical_core(vcpu[i], core[i]);
}
\`\`\`

## Conclusion

Optimizing hypervisor performance requires a multi-faceted approach addressing memory, I/O, and CPU scheduling. With these techniques, you can achieve near-native performance in virtualized environments.

## References

1. Smith, J. & Johnson, K. (2024). "Hypervisor Performance Optimization"
2. Technical Report TR-2024-001, University of Technology`,

};

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