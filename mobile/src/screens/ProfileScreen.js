import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ScrollView
} from 'react-native';

import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useAuth } from '../context/AuthContext';

const BELTS = ['white', 'blue', 'purple', 'brown', 'black'];

export default function ProfileScreen() {
  const { user, logout } = useAuth();

  const [selectedBelt, setSelectedBelt] = useState(
    user?.belt || 'white'
  );

  const BELT_COLORS = {
    white: '#ffffff',
    blue: '#1E90FF',
    purple: '#8B5CF6',
    brown: '#92400E',
    black: '#1a1a1a'
  };

  const handleLogout = () => {
    Alert.alert(
      'Sign Out',
      'Are you sure you want to sign out?',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Sign Out', style: 'destructive', onPress: logout }
      ]
    );
  };

  return (
    <LinearGradient colors={['#0a0a0f', '#12121a']} style={styles.gradient}>
      <ScrollView style={styles.scroll}>
        <View style={styles.header}>
          <View
            style={[
              styles.avatar,
              {
                borderColor:
                  BELT_COLORS[selectedBelt] || '#C8102E'
              }
            ]}
          >
            <Text style={styles.avatarText}>
              {user?.name
                ?.split(' ')
                .map(n => n[0])
                .join('')
                .toUpperCase()
                .slice(0, 2)}
            </Text>
          </View>

          <Text style={styles.userName}>{user?.name}</Text>
          <Text style={styles.userEmail}>{user?.email}</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Belt Rank</Text>

          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
          >
            {BELTS.map(belt => (
              <TouchableOpacity
                key={belt}
                onPress={() => setSelectedBelt(belt)}
                style={[
                  styles.beltBtn,
                  {
                    backgroundColor:
                      selectedBelt === belt
                        ? BELT_COLORS[belt]
                        : '#12121a'
                  },
                  selectedBelt === belt &&
                    styles.beltBtnActive
                ]}
              >
                <Text
                  style={[
                    styles.beltText,
                    {
                      color:
                        selectedBelt === belt &&
                        belt !== 'white'
                          ? '#fff'
                          : selectedBelt === belt
                          ? '#000'
                          : '#6b7280'
                    }
                  ]}
                >
                  {belt.charAt(0).toUpperCase() +
                    belt.slice(1)}
                </Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>
            Account Info
          </Text>

          {[
            {
              icon: 'person-outline',
              label: 'Full Name',
              value: user?.name
            },
            {
              icon: 'mail-outline',
              label: 'Email',
              value: user?.email
            },
            {
              icon: 'calendar-outline',
              label: 'Member Since',
              value: user?.created_at
                ? new Date(
                    user.created_at
                  ).toLocaleDateString()
                : 'N/A'
            }
          ].map(item => (
            <View
              key={item.label}
              style={styles.infoRow}
            >
              <Ionicons
                name={item.icon}
                size={20}
                color="#C8102E"
              />

              <View style={styles.infoContent}>
                <Text style={styles.infoLabel}>
                  {item.label}
                </Text>

                <Text style={styles.infoValue}>
                  {item.value}
                </Text>
              </View>
            </View>
          ))}
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>About</Text>

          <View style={styles.aboutCard}>
            <Text style={styles.aboutTitle}>
              BJJ Analyzer v1.0
            </Text>

            <Text style={styles.aboutText}>
              AI-powered Brazilian Jiu-Jitsu
              match analysis platform. Uses
              computer vision and pose
              estimation to detect positions,
              track athletes, and generate
              intelligent match summaries.
            </Text>
          </View>
        </View>

        <TouchableOpacity
          style={styles.logoutBtn}
          onPress={handleLogout}
        >
          <Ionicons
            name="log-out-outline"
            size={20}
            color="#C8102E"
          />

          <Text style={styles.logoutText}>
            Sign Out
          </Text>
        </TouchableOpacity>
      </ScrollView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  gradient: {
    flex: 1
  },

  scroll: {
    flex: 1,
    paddingTop: 60
  },

  header: {
    alignItems: 'center',
    paddingHorizontal: 24,
    marginBottom: 40
  },

  avatar: {
    width: 90,
    height: 90,
    borderRadius: 45,
    backgroundColor: '#1a1a26',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 3,
    marginBottom: 16
  },

  avatarText: {
    fontSize: 32,
    fontWeight: '900',
    color: '#fff'
  },

  userName: {
    fontSize: 24,
    fontWeight: '900',
    color: '#fff',
    marginBottom: 4
  },

  userEmail: {
    fontSize: 14,
    color: '#6b7280'
  },

  section: {
    paddingHorizontal: 24,
    marginBottom: 28
  },

  sectionTitle: {
    fontSize: 16,
    fontWeight: '800',
    color: '#fff',
    marginBottom: 16,
    letterSpacing: 0.5
  },

  beltBtn: {
    paddingHorizontal: 18,
    paddingVertical: 10,
    borderRadius: 20,
    marginRight: 10,
    borderWidth: 2,
    borderColor: 'transparent'
  },

  beltBtnActive: {
    borderColor: '#C8102E'
  },

  beltText: {
    fontWeight: '700',
    fontSize: 13
  },

  infoRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    backgroundColor: '#1a1a26',
    borderRadius: 14,
    padding: 16,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#2a2a3a'
  },

  infoContent: {
    flex: 1
  },

  infoLabel: {
    fontSize: 11,
    color: '#6b7280',
    fontWeight: '600',
    letterSpacing: 1,
    marginBottom: 2
  },

  infoValue: {
    fontSize: 15,
    color: '#fff',
    fontWeight: '600'
  },

  aboutCard: {
    backgroundColor: '#1a1a26',
    borderRadius: 14,
    padding: 20,
    borderWidth: 1,
    borderColor: '#2a2a3a'
  },

  aboutTitle: {
    fontSize: 16,
    color: '#fff',
    fontWeight: '800',
    marginBottom: 10
  },

  aboutText: {
    fontSize: 14,
    color: '#9ca3af',
    lineHeight: 22
  },

  logoutBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    margin: 24,
    padding: 18,
    backgroundColor: '#C8102E10',
    borderRadius: 14,
    borderWidth: 1,
    borderColor: '#C8102E40',
    marginBottom: 40
  },

  logoutText: {
    color: '#C8102E',
    fontSize: 16,
    fontWeight: '800'
  }
});