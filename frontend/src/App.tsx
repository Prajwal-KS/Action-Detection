import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ProcessProvider } from './context/ProcessContext';
// import { AuthProvider } from './context/AuthContext';
import { ThemeProvider } from './context/ThemeContext';

import Navbar from './components/Navbar';
import ProcessPage from './pages/ProcessPage';
import ModelDetails from './pages/ModelDetails';
import FAQ from './pages/FAQ';
import LoginPage from './pages/LoginPage';
import SignupPage from './pages/SignupPage';

function App() {
  return (
    <Router>

        <ThemeProvider>
        <ProcessProvider>
        
          <Routes>
            {/* <Route path="/" element={<LoginPage />} />
            <Route path="/signup" element={<SignupPage />} /> */}
            
            <Route 
              path="/" 
              element={

                  <div className="min-h-screen transition-colors duration-200 dark:bg-gray-900">
                  <Navbar />
                  <div className="container mx-auto px-4 py-8">
                    <ProcessPage />
                  </div>
                </div>

              } 
            />
            
            <Route path="/model-details" element={

                <div className="min-h-screen transition-colors duration-200 dark:bg-gray-900">
                  <Navbar />
                  <div className="container mx-auto px-4 py-8">
                    <ModelDetails />
                  </div>
                </div>

            } />
            
            <Route path="/faq" element={

                <div className="min-h-screen transition-colors duration-200 dark:bg-gray-900">
                  <Navbar />
                  <div className="container mx-auto px-4 py-8">
                    <FAQ />
                  </div>
                </div>

            } />
          </Routes>
        </ProcessProvider>
        </ThemeProvider>

    </Router>
  );
}

export default App;