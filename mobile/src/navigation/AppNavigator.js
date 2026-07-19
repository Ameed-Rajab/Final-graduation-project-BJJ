import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import { View, ActivityIndicator, StyleSheet } from 'react-native';

import { useAuth } from '../context/AuthContext';
import LoginScreen from '../screens/LoginScreen';
import RegisterScreen from '../screens/RegisterScreen';
import HomeScreen from '../screens/HomeScreen';
import ResultsScreen from '../screens/ResultsScreen';
import HistoryScreen from '../screens/HistoryScreen';
import ProfileScreen from '../screens/ProfileScreen';

const Stack = createNativeStackNavigator();
const Tab = createBottomTabNavigator();

function MainTabs() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarStyle: {
          backgroundColor: '#12121a',
          borderTopColor: '#2a2a3a',
          borderTopWidth: 1,
          paddingBottom: 8,
          paddingTop: 8,
          height: 70
        },
        tabBarActiveTintColor: '#C8102E',
        tabBarInactiveTintColor: '#6b7280',
        tabBarIcon: ({ focused, color, size }) => {
          const icons = {
            Home: focused ? 'home' : 'home-outline',
            History: focused ? 'time' : 'time-outline',
            Profile: focused ? 'person' : 'person-outline'
          };
          return <Ionicons name={icons[route.name]} size={22} color={color} />;
        }
      })}
    >
      <Tab.Screen name="Home" component={HomeScreen} />
      <Tab.Screen name="History" component={HistoryScreen} />
      <Tab.Screen name="Profile" component={ProfileScreen} />
    </Tab.Navigator>
  );
}

export default function AppNavigator() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <View style={styles.loading}>
        <ActivityIndicator size={36} color="#C8102E" />
      </View>
    );
  }

  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        {user ? (
          <>
            <Stack.Screen name="Main" component={MainTabs} />
            <Stack.Screen name="Results" component={ResultsScreen} />
          </>
        ) : (
          <>
            <Stack.Screen name="Login" component={LoginScreen} />
            <Stack.Screen name="Register" component={RegisterScreen} />
          </>
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  loading: { flex: 1, backgroundColor: '#0a0a0f', justifyContent: 'center', alignItems: 'center' }
});
