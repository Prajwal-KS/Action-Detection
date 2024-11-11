import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import ProcessPage from './pages/ProcessPage';
import ModelDetails from './pages/ModelDetails';
import Blog from './pages/Blog';
import { ThemeProvider } from './context/ThemeContext';
import { ProcessProvider } from './context/ProcessContext';

function App() {
  return (
    <BrowserRouter>
      <ThemeProvider>
        <ProcessProvider>
          <div className="min-h-screen transition-colors duration-200 dark:bg-gray-900">
            <Navbar />
            <div className="container mx-auto px-4 py-8">
              <Routes>
                <Route path="/" element={<ProcessPage />} />
                <Route path="/model-details" element={<ModelDetails />} />
                <Route path="/blog" element={<Blog />} />
              </Routes>
            </div>
          </div>
        </ProcessProvider>
      </ThemeProvider>
    </BrowserRouter>
  );
}

export default App;