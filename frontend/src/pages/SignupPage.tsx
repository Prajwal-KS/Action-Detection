import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Mail, LockKeyhole, UserPlus, Moon, Sun } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

const SignupPage = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const { signUp } = useAuth();
  const { darkMode, toggleDarkMode } = useTheme();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      setError('');
      await signUp(email, password);
    } catch (err: any) {
      setError(err.message || 'Failed to create an account');
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 dark:bg-gray-900 py-12 px-4 sm:px-6 lg:px-8">
      {/* Toggle Button at Top Right */}
      <div className="absolute top-4 right-4">
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
      </div>

      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900 dark:text-white">
            Create your account
          </h2>
        </div>

        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          {error && (
            <div className="bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 p-3 rounded-md">
              {error}
            </div>
          )}

          <div className="rounded-md shadow-sm -space-y-px">
            <div>
              <label htmlFor="email" className="sr-only">Email address</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center">
                  <Mail className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  id="email"
                  name="email"
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="appearance-none block w-full px-10 py-2 border border-gray-300 dark:border-gray-700 
                           placeholder-gray-500 dark:placeholder-gray-400 
                           text-gray-900 dark:text-white 
                           rounded-t-md focus:outline-none focus:ring-primary-500 
                           focus:border-primary-500 focus:z-10 sm:text-sm
                           dark:bg-gray-800"
                  placeholder="Email address"
                />
              </div>
            </div>
            <div>
              <label htmlFor="password" className="sr-only">Password</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center">
                  <LockKeyhole className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  id="password"
                  name="password"
                  type="password"
                  required
                  minLength={6}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="appearance-none block w-full px-10 py-2 border border-gray-300 dark:border-gray-700 
                           placeholder-gray-500 dark:placeholder-gray-400 
                           text-gray-900 dark:text-white 
                           rounded-b-md focus:outline-none focus:ring-primary-500 
                           focus:border-primary-500 focus:z-10 sm:text-sm
                           dark:bg-gray-800"
                  placeholder="Password"
                />
              </div>
            </div>
          </div>

          <div>
            <button
              type="submit"
              className="group relative w-full flex justify-center py-2 px-4 border border-transparent 
                       text-sm font-medium rounded-md text-white bg-primary-600 
                       hover:bg-primary-700 focus:outline-none focus:ring-2 
                       focus:ring-offset-2 focus:ring-primary-500"
            >
              <span className="absolute left-0 inset-y-0 flex items-center pl-3">
                <UserPlus className="h-5 w-5 text-primary-500 group-hover:text-primary-400" />
              </span>
              Sign up
            </button>
          </div>

          <div className="text-center">
            <Link
              to="/"
              className="font-medium text-primary-600 hover:text-primary-500 dark:text-primary-400"
            >
              Already have an account? Sign in
            </Link>
          </div>
        </form>
      </div>
    </div>
  );
};

export default SignupPage;
