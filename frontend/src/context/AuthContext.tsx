import React, { createContext, useContext, useState, useEffect } from 'react';
import { 
  createUserWithEmailAndPassword, 
  signInWithEmailAndPassword, 
  signOut, 
  onAuthStateChanged,
  User as FirebaseUser
} from 'firebase/auth';
import { auth } from '../config/firebase';
import { useNavigate } from 'react-router-dom';

// Define User type
interface User {
  id: string;
  email: string | null;
}

// Define AuthContext type
interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string) => Promise<void>;
  logout: () => void;
  loading: boolean;
}

// Create context
const AuthContext = createContext<AuthContextType | undefined>(undefined);

// AuthProvider component
export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  // Listen for auth state changes
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (firebaseUser) => {
      if (firebaseUser) {
        // Transform Firebase user to our User type
        const userData: User = {
          id: firebaseUser.uid,
          email: firebaseUser.email
        };
        setUser(userData);
        setIsAuthenticated(true);
      } else {
        setUser(null);
        setIsAuthenticated(false);
      }
      setLoading(false);
    });

    // Cleanup subscription
    return () => unsubscribe();
  }, []);

  // Sign In method
  const signIn = async (email: string, password: string) => {
    try {
      const userCredential = await signInWithEmailAndPassword(auth, email, password);
      // Navigation handled by onAuthStateChanged
      navigate('/process-page');
    } catch (error: any) {
      console.error('Login error', error);
      
      // Handle specific Firebase error codes
      switch(error.code) {
        case 'auth/wrong-password':
          throw new Error('Incorrect password');
        case 'auth/user-not-found':
          throw new Error('No user found with this email');
        case 'auth/invalid-email':
          throw new Error('Invalid email address');
        default:
          throw new Error('Login failed');
      }
    }
  };

  // Sign Up method
  const signUp = async (email: string, password: string) => {
    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      // Navigation handled by onAuthStateChanged
      navigate('/process-page');
    } catch (error: any) {
      console.error('Signup error', error);
      
      // Handle specific Firebase error codes
      switch(error.code) {
        case 'auth/email-already-in-use':
          throw new Error('Email already in use');
        case 'auth/invalid-email':
          throw new Error('Invalid email address');
        case 'auth/weak-password':
          throw new Error('Password is too weak');
        default:
          throw new Error('Signup failed');
      }
    }
  };

  // Logout method
  const logout = async () => {
    try {
      await signOut(auth);
      navigate('/');
    } catch (error) {
      console.error('Logout error', error);
    }
  };

  // Context value
  const contextValue = {
    user,
    isAuthenticated,
    signIn,
    signUp,
    logout,
    loading
  };

  // Render children with loading state
  return (
    <AuthContext.Provider value={contextValue}>
      {loading ? <div>Loading...</div> : children}
    </AuthContext.Provider>
  );
};

// Custom hook to use auth context
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};