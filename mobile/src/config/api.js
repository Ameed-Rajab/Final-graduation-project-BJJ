import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

export const BASE_URL = 'http://192.168.1.60:5000';
const api = axios.create({
  baseURL: `${BASE_URL}/api`,
  timeout: 120000,
});

api.interceptors.request.use(async (config) => {
  const token = await AsyncStorage.getItem('bjj_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      await AsyncStorage.removeItem('bjj_token');
    }
    return Promise.reject(error);
  }
);

export default api;