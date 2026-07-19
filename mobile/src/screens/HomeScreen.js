import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Alert,
  ActivityIndicator,
  Animated,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useAuth } from '../context/AuthContext';
import api from '../config/api';
import AsyncStorage from '@react-native-async-storage/async-storage';

export default function HomeScreen({ navigation }) {
  const { user } = useAuth();
  const [uploading, setUploading] = useState(false);
  const [title, setTitle] = useState('');
  const pulseAnim = useRef(new Animated.Value(1)).current;

  const BELT_COLORS = {
    white: '#ffffff',
    blue: '#1E90FF',
    purple: '#8B5CF6',
    brown: '#92400E',
    black: '#374151',
  };

  const startPulse = () => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.08,
          duration: 800,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 800,
          useNativeDriver: true,
        }),
      ])
    ).start();
  };

  const stopPulse = () => {
    pulseAnim.stopAnimation();
    Animated.timing(pulseAnim, {
      toValue: 1,
      duration: 200,
      useNativeDriver: true,
    }).start();
  };

  const uploadMedia = async (uri, type, mediaType, filename) => {
    const sessionTitle =
      title.trim() || `BJJ Session ${new Date().toLocaleDateString()}`;

    const formData = new FormData();

    formData.append('file', {
      uri,
      name: filename || `upload.${mediaType === 'video' ? 'mp4' : 'jpg'}`,
      type: type || (mediaType === 'video' ? 'video/mp4' : 'image/jpeg'),
    });

    formData.append('title', sessionTitle);
    formData.append('media_type', mediaType);

    setUploading(true);
    startPulse();

    try {
      const token = await AsyncStorage.getItem('bjj_token');

      if (!token) {
        Alert.alert('Login Required', 'No token found. Please login again.');
        return;
      }

      const response = await api.post('/analysis/upload', formData, {
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('UPLOAD SUCCESS:', response.data);

      const { analysis_id } = response.data;

      navigation.navigate('Results', {
        analysisId: analysis_id,
      });
    } catch (error) {
      console.log('UPLOAD ERROR STATUS:', error.response?.status);
      console.log('UPLOAD ERROR DATA:', error.response?.data);
      console.log('UPLOAD ERROR MESSAGE:', error.message);

      Alert.alert(
        'Upload Failed',
        error.response?.data?.error ||
          error.response?.data?.msg ||
          error.message ||
          'Something went wrong'
      );
    } finally {
      setUploading(false);
      stopPulse();
    }
  };

  const pickVideo = async () => {
    try {
      const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();

      if (!permission.granted) {
        Alert.alert('Permission Required', 'Please allow access to your media library');
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['videos'],
        allowsEditing: false,
        quality: 1,
      });

      console.log('VIDEO PICK RESULT:', result);

      if (!result.canceled && result.assets?.[0]) {
        const asset = result.assets[0];

        await uploadMedia(
          asset.uri,
          asset.mimeType || 'video/mp4',
          'video',
          asset.fileName || 'match.mp4'
        );
      }
    } catch (error) {
      console.log('PICK VIDEO ERROR:', error);
      Alert.alert('Error', error.message || 'Could not pick video');
    }
  };

  const pickImage = async () => {
    try {
      const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();

      if (!permission.granted) {
        Alert.alert('Permission Required', 'Please allow access to your media library');
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        allowsEditing: true,
        quality: 1,
      });

      console.log('IMAGE PICK RESULT:', result);

      if (!result.canceled && result.assets?.[0]) {
        const asset = result.assets[0];

        await uploadMedia(
          asset.uri,
          asset.mimeType || 'image/jpeg',
          'image',
          asset.fileName || 'frame.jpg'
        );
      }
    } catch (error) {
      console.log('PICK IMAGE ERROR:', error);
      Alert.alert('Error', error.message || 'Could not pick image');
    }
  };

  const recordVideo = async () => {
  try {
    const cameraPermission = await ImagePicker.requestCameraPermissionsAsync();

    if (!cameraPermission.granted) {
      Alert.alert(
        'Camera Permission Required',
        'Go to Settings and allow camera access.'
      );
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ['videos'],
      allowsEditing: false,
      videoMaxDuration: 300,
      quality: 0.8,
    });

    console.log('CAMERA RESULT:', result);

    if (!result.canceled && result.assets?.[0]) {
      const asset = result.assets[0];

      await uploadMedia(
        asset.uri,
        asset.mimeType || 'video/mp4',
        'video',
        asset.fileName || 'recorded.mp4'
      );
    }
  } catch (error) {
    console.log('CAMERA ERROR:', error);
    Alert.alert('Camera Error', error.message || 'Camera did not open');
  }
};
  const beltColor = BELT_COLORS[user?.belt] || '#ffffff';

  return (
    <LinearGradient colors={['#0a0a0f', '#12121a']} style={styles.gradient}>
      <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <View>
            <Text style={styles.greeting}>Welcome back,</Text>
            <Text style={styles.username}>{user?.name || 'Athlete'}</Text>
          </View>

          <View style={[styles.beltBadge, { backgroundColor: beltColor }]}>
            <Text
              style={[
                styles.beltText,
                { color: user?.belt === 'white' ? '#000' : '#fff' },
              ]}
            >
              {(user?.belt || 'white').toUpperCase()}
            </Text>
          </View>
        </View>

        <Animated.View style={[styles.heroCard, { transform: [{ scale: pulseAnim }] }]}>
          <LinearGradient
            colors={['#C8102E', '#8B0000']}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.heroGradient}
          >
            <Text style={styles.heroTitle}>AI Match Analysis</Text>

            <Text style={styles.heroSubtitle}>
              Upload your BJJ session and get instant AI-powered insights,
              position tracking, and match summaries.
            </Text>

            <View style={styles.heroBadges}>
              {['Pose Estimation', 'Position Detection', 'Match Summary'].map((b) => (
                <View key={b} style={styles.badge}>
                  <Text style={styles.badgeText}>{b}</Text>
                </View>
              ))}
            </View>
          </LinearGradient>
        </Animated.View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Analyze New Session</Text>

          {uploading && (
            <View style={styles.uploadingCard}>
              <ActivityIndicator color="#C8102E" size={36} />
              <Text style={styles.uploadingText}>Uploading & processing...</Text>
              <Text style={styles.uploadingSubtext}>
                AI is analyzing your BJJ session
              </Text>
            </View>
          )}

          {!uploading && (
            <View style={styles.actionGrid}>
              <TouchableOpacity style={styles.actionCard} onPress={pickVideo}>
                <LinearGradient colors={['#1a1a26', '#22223a']} style={styles.actionGradient}>
                  <View style={[styles.actionIcon, { backgroundColor: '#C8102E20' }]}>
                    <Ionicons name="videocam" size={28} color="#C8102E" />
                  </View>
                  <Text style={styles.actionTitle}>Upload Video</Text>
                  <Text style={styles.actionDesc}>From gallery</Text>
                </LinearGradient>
              </TouchableOpacity>

              <TouchableOpacity style={styles.actionCard} onPress={recordVideo}>
                <LinearGradient colors={['#1a1a26', '#22223a']} style={styles.actionGradient}>
                  <View style={[styles.actionIcon, { backgroundColor: '#FF3B5C20' }]}>
                    <Ionicons name="radio-button-on" size={28} color="#FF3B5C" />
                  </View>
                  <Text style={styles.actionTitle}>Record</Text>
                  <Text style={styles.actionDesc}>Live session</Text>
                </LinearGradient>
              </TouchableOpacity>

              
            </View>
          )}
        </View>

       

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Quick Actions</Text>

          <TouchableOpacity
            style={styles.quickAction}
            onPress={() => navigation.navigate('History')}
          >
            <View style={styles.quickActionLeft}>
              <Ionicons name="time-outline" size={22} color="#C8102E" />
              <Text style={styles.quickActionText}>View Analysis History</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color="#6b7280" />
          </TouchableOpacity>
        </View>
      </ScrollView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  gradient: { flex: 1 },
  scroll: { flex: 1, paddingTop: 60 },

  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 24,
    marginBottom: 24,
  },

  greeting: {
    fontSize: 14,
    color: '#6b7280',
    letterSpacing: 1,
  },

  username: {
    fontSize: 22,
    fontWeight: '900',
    color: '#fff',
    marginTop: 2,
  },

  beltBadge: {
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 20,
  },

  beltText: {
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 2,
  },

  heroCard: {
    marginHorizontal: 24,
    borderRadius: 20,
    overflow: 'hidden',
    marginBottom: 32,
  },

  heroGradient: {
    padding: 28,
  },

  heroTitle: {
    fontSize: 26,
    fontWeight: '900',
    color: '#fff',
    marginBottom: 10,
  },

  heroSubtitle: {
    fontSize: 14,
    color: '#ffcccc',
    lineHeight: 22,
    marginBottom: 20,
  },

  heroBadges: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },

  badge: {
    backgroundColor: '#ffffff20',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },

  badgeText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },

  section: {
    paddingHorizontal: 24,
    marginBottom: 24,
  },

  sectionTitle: {
    fontSize: 18,
    fontWeight: '800',
    color: '#fff',
    marginBottom: 16,
    letterSpacing: 0.5,
  },

  uploadingCard: {
    backgroundColor: '#1a1a26',
    borderRadius: 16,
    padding: 32,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#C8102E30',
  },

  uploadingText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
    marginTop: 16,
  },

  uploadingSubtext: {
    color: '#6b7280',
    fontSize: 13,
    marginTop: 6,
  },

  actionGrid: {
    flexDirection: 'row',
    gap: 12,
  },

  actionCard: {
    flex: 1,
    borderRadius: 16,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#2a2a3a',
  },

  actionGradient: {
    padding: 20,
    alignItems: 'center',
  },

  actionIcon: {
    width: 54,
    height: 54,
    borderRadius: 27,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
  },

  actionTitle: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '800',
    marginBottom: 4,
  },

  actionDesc: {
    color: '#6b7280',
    fontSize: 11,
  },

  statsRow: {
    flexDirection: 'row',
    paddingHorizontal: 24,
    gap: 12,
    marginBottom: 24,
  },

  statCard: {
    flex: 1,
    backgroundColor: '#1a1a26',
    borderRadius: 14,
    padding: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#2a2a3a',
  },

  statValue: {
    fontSize: 22,
    fontWeight: '900',
    color: '#fff',
    marginTop: 8,
  },

  statLabel: {
    fontSize: 11,
    color: '#6b7280',
    marginTop: 4,
    letterSpacing: 1,
  },

  quickAction: {
    backgroundColor: '#1a1a26',
    borderRadius: 14,
    padding: 18,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#2a2a3a',
  },

  quickActionLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },

  quickActionText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '600',
  },
});