import React from 'react';
import { Calendar, User, Clock } from 'lucide-react';

const Blog = () => {
  const blogPosts = [
    {
      title: 'Understanding Action Detection in Video Analysis',
      excerpt: 'Deep dive into how AI models detect and classify actions in video streams...',
      author: 'Dr. Sarah Chen',
      date: '2024-03-15',
      readTime: '5 min read',
      image: 'https://images.unsplash.com/photo-1633613286991-611fe299c4be?auto=format&fit=crop&q=80&w=800',
    },
    {
      title: 'The Evolution of YOLO in Computer Vision',
      excerpt: 'Exploring the journey of YOLO from its inception to current state-of-the-art implementations...',
      author: 'James Wilson',
      date: '2024-03-10',
      readTime: '8 min read',
      image: 'https://images.unsplash.com/photo-1620712943543-bcc4688e7485?auto=format&fit=crop&q=80&w=800',
    },
    {
      title: 'Real-world Applications of Action Detection',
      excerpt: 'Case studies and practical applications of action detection in various industries...',
      author: 'Maria Rodriguez',
      date: '2024-03-05',
      readTime: '6 min read',
      image: 'https://images.unsplash.com/photo-1485827404703-89b55fcc595e?auto=format&fit=crop&q=80&w=800',
    },
  ];

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-8">
          Latest Articles
        </h1>

        <div className="space-y-8">
          {blogPosts.map((post, index) => (
            <article
              key={index}
              className="bg-gray-50 dark:bg-gray-700 rounded-lg overflow-hidden shadow-md transition-transform duration-300 hover:transform hover:scale-[1.02]"
            >
              <div className="md:flex">
                <div className="md:flex-shrink-0">
                  <img
                    className="h-48 w-full md:w-48 object-cover"
                    src={post.image}
                    alt={post.title}
                  />
                </div>
                <div className="p-6">
                  <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                    {post.title}
                  </h2>
                  <p className="text-gray-600 dark:text-gray-300 mb-4">
                    {post.excerpt}
                  </p>
                  <div className="flex flex-wrap items-center text-sm text-gray-500 dark:text-gray-400 space-x-4">
                    <div className="flex items-center">
                      <User className="w-4 h-4 mr-1" />
                      <span>{post.author}</span>
                    </div>
                    <div className="flex items-center">
                      <Calendar className="w-4 h-4 mr-1" />
                      <span>{post.date}</span>
                    </div>
                    <div className="flex items-center">
                      <Clock className="w-4 h-4 mr-1" />
                      <span>{post.readTime}</span>
                    </div>
                  </div>
                </div>
              </div>
            </article>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Blog;