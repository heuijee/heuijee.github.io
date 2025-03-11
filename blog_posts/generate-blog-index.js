/**
 * Blog Index Generator
 * 
 * This script scans the blog_posts directory for markdown files,
 * extracts their frontmatter metadata, and generates a blog_index.json file
 * that can be used by the frontend to render the blog list.
 * 
 * Usage:
 * node generate-blog-index.js
 */

const fs = require('fs');
const path = require('path');
const matter = require('gray-matter');

// Configuration
const POSTS_DIR = path.join(__dirname);
const OUTPUT_FILE = path.join(__dirname, 'blog_index.json');
const VALID_EXTENSIONS = ['.md', '.markdown'];

/**
 * Main function to generate the blog index
 */
function generateBlogIndex() {
  console.log('Generating blog index...');
  
  try {
    // Get all markdown files
    const files = getMarkdownFiles(POSTS_DIR);
    
    // Extract metadata from each file
    const postsData = files.map(file => {
      const filePath = path.join(POSTS_DIR, file);
      const fileContent = fs.readFileSync(filePath, 'utf8');
      const { data } = matter(fileContent);
      
      // Skip files without proper frontmatter
      if (!data.title || !data.date) {
        console.warn(`Warning: ${file} is missing required frontmatter (title or date)`);
        return null;
      }
      
      return {
        id: path.basename(file, path.extname(file)),
        path: `blog_posts/${file}`,
        title: data.title,
        date: data.date instanceof Date ? data.date.toISOString().split('T')[0] : data.date,
        author: data.author || 'Heuijee Yun',
        excerpt: data.excerpt || ''
      };
    }).filter(Boolean); // Remove null entries
    
    // Sort posts by date (newest first)
    postsData.sort((a, b) => new Date(b.date) - new Date(a.date));
    
    // Write to output file
    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(postsData, null, 2));
    console.log(`Successfully generated blog index with ${postsData.length} posts`);
    console.log(`Output file: ${OUTPUT_FILE}`);
    
    return true;
  } catch (error) {
    console.error('Error generating blog index:', error);
    return false;
  }
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

// Execute the generator
generateBlogIndex(); 