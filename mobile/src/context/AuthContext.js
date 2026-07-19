import React, { createContext, useContext, useState, useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import api from '../config/api';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStoredAuth();
  }, []);

  const loadStoredAuth = async () => {
    try {
      const storedToken = await AsyncStorage.getItem('bjj_token');
      const storedUser = await AsyncStorage.getItem('bjj_user');
      if (storedToken && storedUser) {
        setToken(storedToken);
        setUser(JSON.parse(storedUser));
      }
    } catch (e) {
      console.error('Auth load error:', e);
    } finally {
      setLoading(false);
    }
  };

  const login = async (email, password) => {
    const response = await api.post('/auth/login', { email, password });
    const { token: newToken, user: newUser } = response.data;
    await AsyncStorage.setItem('bjj_token', newToken);
    await AsyncStorage.setItem('bjj_user', JSON.stringify(newUser));
    setToken(newToken);
    setUser(newUser);
    return response.data;
  };

  const register = async (name, email, password, belt = 'white') => {
    const response = await api.post('/auth/register', { name, email, password, belt });
    const { token: newToken, user: newUser } = response.data;
    await AsyncStorage.setItem('bjj_token', newToken);
    await AsyncStorage.setItem('bjj_user', JSON.stringify(newUser));
    setToken(newToken);
    setUser(newUser);
    return response.data;
  };

  const logout = async () => {
    await AsyncStorage.removeItem('bjj_token');
    await AsyncStorage.removeItem('bjj_user');
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, token, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
