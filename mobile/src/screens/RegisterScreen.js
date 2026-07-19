import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Alert,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';

import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../context/AuthContext';

export default function RegisterScreen({ navigation }) {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  const { register } = useAuth();

  const handleRegister = async () => {
    console.log('register button clicked');

    if (!name.trim() || !email.trim() || !password.trim()) {
      Alert.alert('Error', 'Please fill in all fields');
      return;
    }

    if (password.length < 6) {
      Alert.alert('Error', 'Password must be at least 6 characters');
      return;
    }

    setLoading(true);

    try {
      console.log('before register');

      const result = await register(
        name.trim(),
        email.trim(),
        password
      );

      console.log('register success', result);

      Alert.alert('Success', 'Account created successfully');

      navigation.navigate('Login');
    } catch (error) {
      console.log('register error', error);
      console.log('register error response', error?.response?.data);

      Alert.alert(
        'Registration Failed',
        error.response?.data?.error || 'Something went wrong'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <LinearGradient
      colors={['#0a0a0f', '#12121a', '#0a0a0f']}
      style={styles.gradient}
    >
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={{ flex: 1 }}
      >
        <ScrollView
          contentContainerStyle={styles.container}
          keyboardShouldPersistTaps="handled"
        >
          <View style={styles.header}>
            <View style={styles.logoCircle}>
              <Text style={styles.logoText}>BJJ</Text>
            </View>

            <Text style={styles.title}>Create Account</Text>

            <Text style={styles.subtitle}>
              Join the AI revolution in BJJ
            </Text>
          </View>

          <View style={styles.card}>
            <Text style={styles.label}>Full Name</Text>

            <TextInput
              style={styles.input}
              placeholder="John Silva"
              placeholderTextColor="#4a4a5a"
              value={name}
              onChangeText={setName}
            />

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
              placeholder="Min. 6 characters"
              placeholderTextColor="#4a4a5a"
              value={password}
              onChangeText={setPassword}
              secureTextEntry
            />

            <TouchableOpacity
              style={[
                styles.btnPrimary,
                loading && { opacity: 0.7 }
              ]}
              onPress={handleRegister}
              disabled={loading}
              activeOpacity={0.85}
            >
              {loading ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.btnText}>
                  Create Account
                </Text>
              )}
            </TouchableOpacity>

            <TouchableOpacity
              onPress={() => navigation.navigate('Login')}
            >
              <Text style={styles.linkText}>
                Already have an account?{' '}
                <Text style={styles.linkAccent}>
                  Sign In
                </Text>
              </Text>
            </TouchableOpacity>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  gradient: {
    flex: 1,
  },

  container: {
    flexGrow: 1,
    justifyContent: 'center',
    padding: 24,
  },

  header: {
    alignItems: 'center',
    marginBottom: 32,
  },

  logoCircle: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: '#C8102E',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,

    shadowColor: '#C8102E',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 15,
    elevation: 8,
  },

  logoText: {
    fontSize: 20,
    fontWeight: '900',
    color: '#fff',
    letterSpacing: 2,
  },

  title: {
    fontSize: 28,
    fontWeight: '800',
    color: '#fff',
    marginTop: 8,
  },

  subtitle: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 4,
    letterSpacing: 1,
  },

  card: {
    backgroundColor: '#1a1a26',
    borderRadius: 20,
    padding: 28,
    borderWidth: 1,
    borderColor: '#2a2a3a',
  },

  label: {
    fontSize: 13,
    color: '#9ca3af',
    marginBottom: 8,
    fontWeight: '600',
    letterSpacing: 1,
  },

  input: {
    backgroundColor: '#12121a',
    borderRadius: 12,
    padding: 16,
    color: '#fff',
    fontSize: 16,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#2a2a3a',
  },

  btnPrimary: {
    backgroundColor: '#C8102E',
    paddingVertical: 16,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 22,

    shadowColor: '#C8102E',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.4,
    shadowRadius: 8,
    elevation: 6,
  },

  btnText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '800',
    letterSpacing: 1,
  },

  linkText: {
    textAlign: 'center',
    color: '#6b7280',
    fontSize: 14,
  },

  linkAccent: {
    color: '#C8102E',
    fontWeight: '700',
  },
});