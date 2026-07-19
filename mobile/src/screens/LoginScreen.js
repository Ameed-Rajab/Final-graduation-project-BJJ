import React, { useState } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, StyleSheet,
  ScrollView, Alert, ActivityIndicator, KeyboardAvoidingView, Platform
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../context/AuthContext';

export default function LoginScreen({ navigation }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();

  const handleLogin = async () => {
    if (!email.trim() || !password.trim()) {
      Alert.alert('Error', 'Please fill in all fields');
      return;
    }
    setLoading(true);
    try {
      await login(email.trim(), password);
    } catch (error) {
      Alert.alert('Login Failed', error.response?.data?.error || 'Invalid credentials');
    } finally {
      setLoading(false);
    }
  };

  return (
    <LinearGradient colors={['#0a0a0f', '#12121a', '#0a0a0f']} style={styles.gradient}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : 'height'} style={{ flex: 1 }}>
        <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
          <View style={styles.logoContainer}>
            <View style={styles.logoCircle}>
              <Text style={styles.logoText}>BJJ</Text>
            </View>
            <Text style={styles.appTitle}>BJJ Analyzer</Text>
            <Text style={styles.appSubtitle}>AI-Powered Match Intelligence</Text>
          </View>

          <View style={styles.card}>
            <Text style={styles.cardTitle}>Sign In</Text>

            <Text style={styles.label}>Email</Text>
            <TextInput
              style={styles.input}
              placeholder="your@email.com"
              placeholderTextColor="#4a4a5a"
              value={email}
              onChangeText={setEmail}
              autoCapitalize="none"
              keyboardType="email-address"
            />

            <Text style={styles.label}>Password</Text>
            <TextInput
              style={styles.input}
              placeholder="Password"
              placeholderTextColor="#4a4a5a"
              value={password}
              onChangeText={setPassword}
              secureTextEntry
            />

            <TouchableOpacity style={styles.btnPrimary} onPress={handleLogin} disabled={loading}>
              {loading ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.btnText}>Sign In</Text>
              )}
            </TouchableOpacity>

            <TouchableOpacity onPress={() => navigation.navigate('Register')}>
              <Text style={styles.linkText}>
                Don't have an account? <Text style={styles.linkAccent}>Sign Up</Text>
              </Text>
            </TouchableOpacity>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  gradient: { flex: 1 },
  container: { flexGrow: 1, justifyContent: 'center', padding: 24 },
  logoContainer: { alignItems: 'center', marginBottom: 40 },
  logoCircle: {
    width: 80, height: 80, borderRadius: 40,
    backgroundColor: '#C8102E', justifyContent: 'center',
    alignItems: 'center', marginBottom: 16,
    shadowColor: '#C8102E', shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.6, shadowRadius: 20, elevation: 10
  },
  logoText: { fontSize: 24, fontWeight: '900', color: '#fff', letterSpacing: 2 },
  appTitle: { fontSize: 32, fontWeight: '900', color: '#fff', letterSpacing: 1 },
  appSubtitle: { fontSize: 14, color: '#6b7280', marginTop: 4, letterSpacing: 2 },
  card: {
    backgroundColor: '#1a1a26', borderRadius: 20, padding: 28,
    borderWidth: 1, borderColor: '#2a2a3a'
  },
  cardTitle: { fontSize: 22, fontWeight: '800', color: '#fff', marginBottom: 24 },
  label: { fontSize: 13, color: '#9ca3af', marginBottom: 8, fontWeight: '600', letterSpacing: 1 },
  input: {
    backgroundColor: '#12121a', borderRadius: 12, padding: 16,
    color: '#fff', fontSize: 16, marginBottom: 20,
    borderWidth: 1, borderColor: '#2a2a3a'
  },
  btnPrimary: {
    backgroundColor: '#C8102E', borderRadius: 12, padding: 18,
    alignItems: 'center', marginBottom: 20,
    shadowColor: '#C8102E', shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4, shadowRadius: 12, elevation: 8
  },
  btnText: { color: '#fff', fontSize: 16, fontWeight: '800', letterSpacing: 1 },
  linkText: { textAlign: 'center', color: '#6b7280', fontSize: 14 },
  linkAccent: { color: '#C8102E', fontWeight: '700' }
});
