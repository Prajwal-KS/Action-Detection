import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Brain, FileVideo, HelpCircle, Moon, Sun, Rocket, LogOut } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';
import { useAuth } from '../context/AuthContext';

const Navbar = () => {
  const location = useLocation();
  const { darkMode, toggleDarkMode } = useTheme();
  const { logout } = useAuth();
  const isActive = (path: string) => location.pathname === path;

  return (
    <nav className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-lg shadow-lg sticky top-0 z-50 transition-colors duration-200">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-8">
            <Link 
              to="/process-page" 
              className="flex items-center space-x-2 group"
            >
              <FileVideo className="h-7 w-7 text-primary-600 dark:text-primary-400 
                                  group-hover:scale-110 transition-transform duration-200" />
              <span className="text-xl font-bold bg-gradient-to-r from-primary-600 to-primary-400 
                             bg-clip-text text-transparent">
                ActionAI
              </span>
            </Link>
            
            <div className="flex space-x-1">
              <Link
                to="/process-page"
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                  isActive('/process-page') 
                    ? 'bg-primary-50 text-primary-700 dark:bg-primary-900/30 dark:text-primary-300'
                    : 'text-gray-600 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-gray-700/50'
                }`}
              >
                <div className="flex items-center space-x-1">
                  <Rocket className="h-4 w-4" />
                  <span>Process</span>
                </div>
              </Link>
              
              <Link
                to="/model-details"
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                  isActive('/model-details')
                    ? 'bg-primary-50 text-primary-700 dark:bg-primary-900/30 dark:text-primary-300'
                    : 'text-gray-600 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-gray-700/50'
                }`}
              >
                <div className="flex items-center space-x-1">
                  <Brain className="h-4 w-4" />
                  <span>Model Details</span>
                </div>
              </Link>
              
              <Link
                to="/faq"
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                  isActive('/faq')
                    ? 'bg-primary-50 text-primary-700 dark:bg-primary-900/30 dark:text-primary-300'
                    : 'text-gray-600 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-gray-700/50'
                }`}
              >
                <div className="flex items-center space-x-1">
                  <HelpCircle className="h-4 w-4" />
                  <span>FAQ</span>
                </div>
              </Link>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            {/* Dark Mode Toggle Button */}
            <button
              onClick={toggleDarkMode}
              className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 
                     transition-colors duration-200 group"
              aria-label="Toggle dark mode"
            >
              {darkMode ? (
                <Sun className="h-5 w-5 text-yellow-500 group-hover:rotate-180 transition-transform duration-300" />
              ) : (
                <Moon className="h-5 w-5 text-gray-600 group-hover:rotate-180 transition-transform duration-300" />
              )}
            </button>

            {/* Logout Button */}
            <button
              onClick={logout}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 
                text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-500 
                hover:bg-gray-50 dark:hover:bg-gray-700/50`}
            >
              <div className="flex items-center space-x-1">
                <LogOut className="h-4 w-4" />
                <span>Logout</span>
              </div>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
