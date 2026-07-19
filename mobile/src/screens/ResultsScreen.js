import React, { useState, useEffect } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  ActivityIndicator, RefreshControl, Alert
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import api from '../config/api';

const POSITION_COLORS = {
  open_guard: '#4FC3F7',
  closed_guard: '#1E90FF',
  half_guard: '#0288D1',
  mount: '#C8102E',
  side_control: '#FF6B35',
  back: '#FF3B5C',
  standing: '#00C896',
  takedown: '#FFB800',
  turtle: '#F59E0B',
  leg_entanglement: '#8B5CF6',
  finish: '#EF4444',
  unknown: '#6b7280'
};

export default function ResultsScreen({ route, navigation }) {
  const { analysisId } = route.params;
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [polling, setPolling] = useState(true);

  useEffect(() => {
    fetchAnalysis();

    const interval = setInterval(() => {
      if (polling) fetchAnalysis(true);
    }, 3000);

    return () => clearInterval(interval);
  }, [analysisId, polling]);

  const normalizeAnalysis = (rawData) => {
    let raw = rawData;

    if (!raw) return null;

    if (raw.status === 'processing') {
      return { status: 'processing' };
    }

    if (raw.status === 'failed') {
      return {
        status: 'failed',
        summary: raw.summary || 'Analysis failed',
      };
    }

    if (!raw.players && raw.positions_detected) {
      try {
        raw = JSON.parse(raw.positions_detected);
      } catch (e) {
        return { status: 'processing' };
      }
    }

    if (!raw.players || !raw.match_overview || !raw.match_sequence) {
      return { status: 'processing' };
    }

    const topFamily = raw.match_overview.top_position_families?.[0];
    const players = raw.players || [];
    const frames = raw.match_overview.frames_analyzed || raw.source_summary?.frames_used || 0;
    const duration =
      raw.source_summary?.duration_seconds ||
      raw.match_overview.video_duration_seconds ||
      (frames ? frames / 30 : 0);

    return {
      status: 'completed',
      title: 'Player Style Analysis',
      frame_count: frames,
      duration,
      dominant_position: topFamily?.family || 'unknown',
      dominant_label: topFamily?.label || 'Unknown',
      dominant_share: topFamily?.share_percent || 0,
      summary:
        raw.match_sequence.narratives?.join(' ') ||
        `The match mainly stayed in ${topFamily?.label || 'Unknown position'}.`,
      players,
      reliability: raw.reliability,
      match_overview: raw.match_overview,
      match_sequence: raw.match_sequence,
      source_summary: raw.source_summary,
      raw_result: raw,
    };
  };

  const fetchAnalysis = async (silent = false) => {
    if (!silent) setLoading(true);

    try {
      const token = await AsyncStorage.getItem('bjj_token');

      if (!token) {
        setAnalysis({
          status: 'failed',
          summary: 'Login expired. Please sign in again.',
        });
        setPolling(false);
        return;
      }

      const response = await api.get(`/analysis/${analysisId}`);

      let raw = response.data.analysis || response.data;
      raw = raw.predictions || raw;
      
      const normalized = normalizeAnalysis(raw);

      setAnalysis(normalized);

      if (normalized?.status === 'completed' || normalized?.status === 'failed') {
        setPolling(false);
      }

    } catch (error) {
      console.error(error);

      if (error.response?.status === 401) {
        Alert.alert('Login Expired', 'Please sign in again to view this analysis.');
      }

      setAnalysis({
        status: 'failed',
        summary:
          error?.response?.data?.error ||
          error?.response?.data?.msg ||
          error.message ||
          'Analysis failed',
      });
      setPolling(false);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    fetchAnalysis();
  };

  if (loading && !analysis) {
    return (
      <LinearGradient colors={['#0a0a0f', '#12121a']} style={styles.center}>
        <ActivityIndicator size={36} color="#C8102E" />
        <Text style={styles.loadingText}>Loading analysis...</Text>
      </LinearGradient>
    );
  }

  const topFamilies = analysis?.match_overview?.top_position_families || [];
  const longestPhases = analysis?.match_sequence?.longest_phases || [];
  const players = analysis?.players || [];

  return (
    <LinearGradient colors={['#0a0a0f', '#12121a']} style={styles.gradient}>
      <ScrollView
        style={styles.scroll}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#C8102E" />
        }
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.header}>
          <TouchableOpacity onPress={() => navigation.goBack()}>
            <Ionicons name="arrow-back" size={24} color="#fff" />
          </TouchableOpacity>

          <Text style={styles.headerTitle}>Match Analysis</Text>

          <View style={[
            styles.statusBadge,
            { backgroundColor: analysis?.status === 'completed' ? '#00C89620' : '#C8102E20' }
          ]}>
            <Text style={[
              styles.statusText,
              { color: analysis?.status === 'completed' ? '#00C896' : '#C8102E' }
            ]}>
              {analysis?.status?.toUpperCase()}
            </Text>
          </View>
        </View>

        {analysis?.status === 'processing' && (
          <View style={styles.processingCard}>
            <ActivityIndicator color="#C8102E" size={34} />
            <Text style={styles.processingText}>AI is analyzing your session...</Text>
            <Text style={styles.processingSubtext}>Tracking athletes, detecting positions, and building style report</Text>
          </View>
        )}

        {analysis?.status === 'failed' && (
          <View style={styles.errorCard}>
            <Ionicons name="warning" size={42} color="#C8102E" />
            <Text style={styles.errorTitle}>Analysis Failed</Text>
            <Text style={styles.errorText}>{analysis.summary}</Text>
          </View>
        )}

        {analysis?.status === 'completed' && (
          <>
            <View style={styles.titleCard}>
              <Text style={styles.analysisTitle}>{analysis.title}</Text>

              <View style={styles.metaRow}>
                <Ionicons name="film-outline" size={14} color="#6b7280" />
                <Text style={styles.metaText}>{analysis.frame_count} frames</Text>

                <Ionicons name="time-outline" size={14} color="#6b7280" />
                <Text style={styles.metaText}>{analysis.duration.toFixed(1)}s</Text>

                <Ionicons name="shield-checkmark-outline" size={14} color="#6b7280" />
                <Text style={styles.metaText}>
                  {analysis.reliability?.band || 'unknown'} reliability
                </Text>
              </View>
            </View>

            <View style={styles.heroCard}>
              <Text style={styles.sectionLabel}>DOMINANT POSITION</Text>

              <View style={styles.dominantRow}>
                <View
                  style={[
                    styles.bigDot,
                    { backgroundColor: POSITION_COLORS[analysis.dominant_position] || '#C8102E' }
                  ]}
                />
                <View>
                  <Text style={styles.dominantText}>{analysis.dominant_label}</Text>
                  <Text style={styles.subText}>{analysis.dominant_share.toFixed(1)}% of analyzed frames</Text>
                </View>
              </View>
            </View>

            

            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Position Breakdown</Text>

              {topFamilies.map((p, index) => {
                const color = POSITION_COLORS[p.family] || '#6b7280';

                return (
                  <View key={`${p.family}-${index}`} style={styles.breakdownRow}>
                    <View style={styles.positionLeft}>
                      <View style={[styles.positionDot, { backgroundColor: color }]} />
                      <Text style={styles.positionLabel}>{p.label}</Text>
                    </View>

                    <View style={styles.positionRight}>
                      <View style={styles.progressBar}>
                        <View
                          style={[
                            styles.progressFill,
                            {
                              width: `${Math.min(p.share_percent || 0, 100)}%`,
                              backgroundColor: color,
                            }
                          ]}
                        />
                      </View>
                      <Text style={styles.positionPct}>{Number(p.share_percent || 0).toFixed(1)}%</Text>
                    </View>
                  </View>
                );
              })}
            </View>

            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Players Style</Text>

              {players.map((player) => (
                <View key={player.player_id} style={styles.playerCard}>
                  <View style={styles.playerHeader}>
                    <View>
                      <Text style={styles.playerName}>{player.player_name}</Text>
                      <Text style={styles.playerProfile}>{player.primary_profile}</Text>
                    </View>

                    
                  </View>

                  <View style={styles.metricsGrid}>
                    <Metric label="Guard" value={player.metrics?.guard_ratio_percent} />
                    <Metric label="Top" value={player.metrics?.top_ratio_percent} />
                    <Metric label="Turtle" value={player.metrics?.turtle_ratio_percent} />
                    <Metric label="Finish" value={player.metrics?.finish_ratio_percent} />
                  </View>

                  {player.secondary_traits?.length > 0 && (
                    <View style={styles.traitsWrap}>
                      {player.secondary_traits.map((trait, idx) => (
                        <View key={idx} style={styles.traitChip}>
                          <Text style={styles.traitText}>{trait}</Text>
                        </View>
                      ))}
                    </View>
                  )}

                  <View style={styles.counterBox}>
                    <Text style={styles.counterTitle}>Counter Plan</Text>
                    <Text style={styles.counterStyle}>
                      {player.counter_plan?.recommended_style || 'No counter plan available'}
                    </Text>

                    {player.counter_plan?.key_actions?.map((action, idx) => (
                      <View key={idx} style={styles.actionRow}>
                        <Ionicons name="checkmark-circle" size={15} color="#00C896" />
                        <Text style={styles.actionText}>{action}</Text>
                      </View>
                    ))}
                  </View>
                </View>
              ))}
            </View>

            {longestPhases.length > 0 && (
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Longest Phases</Text>

                {longestPhases.map((phase, idx) => (
                  <View key={idx} style={styles.phaseRow}>
                    <View style={styles.phaseIcon}>
                      <Ionicons name="timer-outline" size={18} color="#C8102E" />
                    </View>

                    <View style={styles.phaseContent}>
                      <Text style={styles.phaseTitle}>{phase.label}</Text>
                      <Text style={styles.phaseText}>
                        Frames {phase.start_frame} - {phase.end_frame} • {phase.duration_rows} rows
                      </Text>
                    </View>
                  </View>
                ))}
              </View>
            )}

            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Sequence Details</Text>

              <View style={styles.sequenceCard}>
                <InfoRow label="Segments" value={analysis.match_sequence.segment_count} />
                <InfoRow label="Average phase" value={`${analysis.match_sequence.average_phase_frames} frames`} />
                <InfoRow label="Opening" value={analysis.match_sequence.opening_sequence?.join(' → ') || 'None'} />
                <InfoRow label="Ending" value={analysis.match_sequence.ending_sequence?.join(' → ') || 'None'} />
              </View>
            </View>
          </>
        )}
      </ScrollView>
    </LinearGradient>
  );
}

function Metric({ label, value }) {
  const safeValue = Number(value || 0);

  return (
    <View style={styles.metricBox}>
      <Text style={styles.metricValue}>{safeValue.toFixed(1)}%</Text>
      <Text style={styles.metricLabel}>{label}</Text>
    </View>
  );
}

function InfoRow({ label, value }) {
  return (
    <View style={styles.infoRow}>
      <Text style={styles.infoLabel}>{label}</Text>
      <Text style={styles.infoValue}>{String(value ?? 'None')}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  gradient: { flex: 1 },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  loadingText: { color: '#fff', marginTop: 16, fontSize: 16 },
  scroll: { flex: 1, paddingTop: 60 },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 24,
    marginBottom: 24,
  },
  headerTitle: { fontSize: 18, fontWeight: '900', color: '#fff' },
  statusBadge: { paddingHorizontal: 12, paddingVertical: 5, borderRadius: 20 },
  statusText: { fontSize: 11, fontWeight: '900', letterSpacing: 1 },
  processingCard: {
    margin: 24,
    backgroundColor: '#1a1a26',
    borderRadius: 18,
    padding: 34,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#C8102E30',
  },
  processingText: { color: '#fff', fontSize: 17, fontWeight: '800', marginTop: 16 },
  processingSubtext: { color: '#9ca3af', fontSize: 13, marginTop: 8, textAlign: 'center', lineHeight: 20 },
  titleCard: { paddingHorizontal: 24, marginBottom: 18 },
  analysisTitle: { fontSize: 25, fontWeight: '900', color: '#fff', marginBottom: 10 },
  metaRow: { flexDirection: 'row', alignItems: 'center', flexWrap: 'wrap', gap: 8 },
  metaText: { color: '#9ca3af', fontSize: 13 },
  heroCard: {
    marginHorizontal: 24,
    backgroundColor: '#1a1a26',
    borderRadius: 18,
    padding: 22,
    marginBottom: 18,
    borderWidth: 1,
    borderColor: '#2a2a3a',
  },
  sectionLabel: { fontSize: 11, color: '#9ca3af', fontWeight: '800', letterSpacing: 2, marginBottom: 14 },
  dominantRow: { flexDirection: 'row', alignItems: 'center', gap: 14 },
  bigDot: { width: 18, height: 18, borderRadius: 9 },
  dominantText: { fontSize: 24, fontWeight: '900', color: '#fff' },
  subText: { color: '#9ca3af', fontSize: 13, marginTop: 4 },
  summaryCard: {
    marginHorizontal: 24,
    backgroundColor: '#1a1a26',
    borderRadius: 18,
    padding: 22,
    marginBottom: 24,
    borderWidth: 1,
    borderColor: '#FFB80030',
  },
  summaryHeader: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 14 },
  summaryTitle: { color: '#FFB800', fontSize: 16, fontWeight: '900' },
  summaryText: { color: '#d1d5db', fontSize: 14, lineHeight: 23 },
  section: { paddingHorizontal: 24, marginBottom: 24 },
  sectionTitle: { fontSize: 19, fontWeight: '900', color: '#fff', marginBottom: 16 },
  breakdownRow: {
    backgroundColor: '#1a1a26',
    borderRadius: 14,
    padding: 15,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#2a2a3a',
  },
  positionLeft: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 10 },
  positionDot: { width: 10, height: 10, borderRadius: 5 },
  positionLabel: { color: '#fff', fontSize: 15, fontWeight: '700' },
  positionRight: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  progressBar: { flex: 1, height: 7, backgroundColor: '#2a2a3a', borderRadius: 4, overflow: 'hidden' },
  progressFill: { height: '100%', borderRadius: 4 },
  positionPct: { color: '#9ca3af', fontSize: 13, fontWeight: '800', width: 48, textAlign: 'right' },
  playerCard: {
    backgroundColor: '#1a1a26',
    borderRadius: 18,
    padding: 18,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#2a2a3a',
  },
  playerHeader: { flexDirection: 'row', justifyContent: 'space-between', gap: 12, marginBottom: 16 },
  playerName: { color: '#fff', fontSize: 19, fontWeight: '900' },
  playerProfile: { color: '#9ca3af', fontSize: 13, marginTop: 5, maxWidth: 210, lineHeight: 19 },
  playerBadge: {
    backgroundColor: '#C8102E20',
    borderRadius: 12,
    paddingHorizontal: 10,
    paddingVertical: 6,
    alignSelf: 'flex-start',
  },
  playerBadgeText: { color: '#C8102E', fontSize: 11, fontWeight: '900' },
  metricsGrid: { flexDirection: 'row', gap: 8, marginBottom: 14 },
  metricBox: {
    flex: 1,
    backgroundColor: '#12121a',
    borderRadius: 12,
    paddingVertical: 12,
    alignItems: 'center',
  },
  metricValue: { color: '#fff', fontSize: 15, fontWeight: '900' },
  metricLabel: { color: '#6b7280', fontSize: 11, marginTop: 4 },
  traitsWrap: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginBottom: 14 },
  traitChip: { backgroundColor: '#8B5CF620', paddingHorizontal: 10, paddingVertical: 6, borderRadius: 20 },
  traitText: { color: '#c4b5fd', fontSize: 12, fontWeight: '700' },
  counterBox: {
    backgroundColor: '#12121a',
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: '#00C89620',
  },
  counterTitle: { color: '#00C896', fontSize: 13, fontWeight: '900', marginBottom: 8 },
  counterStyle: { color: '#fff', fontSize: 14, fontWeight: '800', marginBottom: 10, lineHeight: 20 },
  actionRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 8, marginTop: 7 },
  actionText: { color: '#d1d5db', fontSize: 13, lineHeight: 19, flex: 1 },
  phaseRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1a1a26',
    borderRadius: 14,
    padding: 15,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#2a2a3a',
  },
  phaseIcon: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: '#C8102E20',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  phaseContent: { flex: 1 },
  phaseTitle: { color: '#fff', fontSize: 15, fontWeight: '800' },
  phaseText: { color: '#9ca3af', fontSize: 12, marginTop: 4 },
  sequenceCard: {
    backgroundColor: '#1a1a26',
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: '#2a2a3a',
    marginBottom: 40,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 14,
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#2a2a3a',
  },
  infoLabel: { color: '#9ca3af', fontSize: 13, fontWeight: '700' },
  infoValue: { color: '#fff', fontSize: 13, fontWeight: '700', flex: 1, textAlign: 'right' },
  errorCard: {
    margin: 24,
    backgroundColor: '#1a1a26',
    borderRadius: 18,
    padding: 32,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#C8102E30',
  },
  errorTitle: { color: '#fff', fontSize: 20, fontWeight: '900', marginTop: 16 },
  errorText: { color: '#9ca3af', fontSize: 14, marginTop: 8, textAlign: 'center', lineHeight: 22 },
});
