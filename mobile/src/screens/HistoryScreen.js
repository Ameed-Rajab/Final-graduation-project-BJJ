import React, { useState, useEffect } from 'react';
import {
  View, Text, StyleSheet, FlatList, TouchableOpacity,
  ActivityIndicator, Alert, RefreshControl
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import api from '../config/api';

const STATUS_COLORS = {
  completed: '#00C896',
  processing: '#FFB800',
  failed: '#C8102E'
};

export default function HistoryScreen({ navigation }) {
  const [analyses, setAnalyses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const response = await api.get('/history/');
      setAnalyses(response.data.analyses);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    fetchHistory();
  };

  const deleteAnalysis = async (id) => {
    Alert.alert('Delete Analysis', 'Are you sure?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete', style: 'destructive',
        onPress: async () => {
          try {
            await api.delete(`/history/${id}`);
            setAnalyses(prev => prev.filter(a => a.id !== id));
          } catch (e) {
            Alert.alert('Error', 'Could not delete');
          }
        }
      }
    ]);
  };

  const formatDate = (iso) => {
    const date = new Date(iso);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };

  const renderItem = ({ item }) => (
    <TouchableOpacity
      style={styles.card}
      onPress={() => navigation.navigate('Results', { analysisId: item.id })}
      onLongPress={() => deleteAnalysis(item.id)}
      activeOpacity={0.8}
    >
      <View style={styles.cardLeft}>
        <View style={[styles.typeIcon, {
          backgroundColor: item.media_type === 'video' ? '#C8102E20' : '#1E90FF20'
        }]}>
          <Ionicons
            name={item.media_type === 'video' ? 'videocam' : 'image'}
            size={20}
            color={item.media_type === 'video' ? '#C8102E' : '#1E90FF'}
          />
        </View>
      </View>
      <View style={styles.cardContent}>
        <Text style={styles.cardTitle} numberOfLines={1}>{item.title}</Text>
        <View style={styles.cardMeta}>
          <Text style={styles.cardDate}>{formatDate(item.created_at)}</Text>
          {item.dominant_position ? (
            <Text style={styles.cardPosition}>
              {item.dominant_position.replace(/_/g, ' ')}
            </Text>
          ) : null}
        </View>
        {item.status === 'completed' && (
          <View style={styles.scoreRow}>
            <Text style={styles.p1Score}>P1: {item.player1_score?.toFixed(0)}%</Text>
            <Text style={styles.p2Score}>P2: {item.player2_score?.toFixed(0)}%</Text>
          </View>
        )}
      </View>
      <View style={styles.cardRight}>
        <View style={[styles.statusDot, { backgroundColor: STATUS_COLORS[item.status] || '#6b7280' }]} />
        <Text style={[styles.statusText, { color: STATUS_COLORS[item.status] || '#6b7280' }]}>
          {item.status}
        </Text>
        <Ionicons name="chevron-forward" size={16} color="#6b7280" style={{ marginTop: 8 }} />
      </View>
    </TouchableOpacity>
  );

  return (
    <LinearGradient colors={['#0a0a0f', '#12121a']} style={styles.gradient}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Analysis History</Text>
        <Text style={styles.headerSub}>{analyses.length} sessions</Text>
      </View>

      {loading ? (
        <View style={styles.center}>
          <ActivityIndicator size={36} color="#C8102E" />
        </View>
      ) : analyses.length === 0 ? (
        <View style={styles.empty}>
          <Ionicons name="analytics-outline" size={64} color="#2a2a3a" />
          <Text style={styles.emptyTitle}>No Analyses Yet</Text>
          <Text style={styles.emptyText}>Upload your first BJJ session to get started</Text>
        </View>
      ) : (
        <FlatList
          data={analyses}
          renderItem={renderItem}
          keyExtractor={item => item.id.toString()}
          contentContainerStyle={styles.list}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#C8102E" />}
          showsVerticalScrollIndicator={false}
        />
      )}
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  gradient: { flex: 1 },
  header: { paddingTop: 60, paddingHorizontal: 24, paddingBottom: 20 },
  headerTitle: { fontSize: 28, fontWeight: '900', color: '#fff' },
  headerSub: { fontSize: 14, color: '#6b7280', marginTop: 4 },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  list: { padding: 24, gap: 12 },
  card: {
    backgroundColor: '#1a1a26', borderRadius: 16, padding: 18,
    flexDirection: 'row', alignItems: 'center', gap: 14,
    borderWidth: 1, borderColor: '#2a2a3a'
  },
  cardLeft: {},
  typeIcon: { width: 44, height: 44, borderRadius: 22, justifyContent: 'center', alignItems: 'center' },
  cardContent: { flex: 1 },
  cardTitle: { color: '#fff', fontSize: 15, fontWeight: '700', marginBottom: 4 },
  cardMeta: { flexDirection: 'row', gap: 10, alignItems: 'center', marginBottom: 4 },
  cardDate: { color: '#6b7280', fontSize: 12 },
  cardPosition: {
    backgroundColor: '#C8102E20', color: '#C8102E',
    fontSize: 11, paddingHorizontal: 8, paddingVertical: 2,
    borderRadius: 10, fontWeight: '600'
  },
  scoreRow: { flexDirection: 'row', gap: 10 },
  p1Score: { color: '#C8102E', fontSize: 12, fontWeight: '700' },
  p2Score: { color: '#1E90FF', fontSize: 12, fontWeight: '700' },
  cardRight: { alignItems: 'flex-end' },
  statusDot: { width: 8, height: 8, borderRadius: 4, marginBottom: 4 },
  statusText: { fontSize: 10, fontWeight: '700', letterSpacing: 1 },
  empty: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 40 },
  emptyTitle: { color: '#fff', fontSize: 20, fontWeight: '800', marginTop: 16 },
  emptyText: { color: '#6b7280', fontSize: 14, textAlign: 'center', marginTop: 8, lineHeight: 22 }
});
