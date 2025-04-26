import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { StatusBar } from 'react-native';

// Import screens
import HomeScreen from './src/screens/HomeScreen';
import MoodTrackerScreen from './src/screens/MoodTrackerScreen';
import JournalScreen from './src/screens/JournalScreen';
import BreathingExerciseScreen from './src/screens/BreathingExerciseScreen';
import ResourcesScreen from './src/screens/ResourcesScreen';

// Define the type for our navigation stack
export type RootStackParamList = {
  Home: undefined;
  MoodTracker: undefined;
  Journal: undefined;
  BreathingExercise: undefined;
  Resources: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

const App = () => {
  const screens = (
    <>
      <Stack.Screen 
        name="Home" 
        component={HomeScreen}
        options={{ title: 'Mental Health Buddy' }}
      />
      <Stack.Screen 
        name="MoodTracker" 
        component={MoodTrackerScreen}
        options={{ title: 'Mood Tracker' }}
      />
      <Stack.Screen 
        name="Journal" 
        component={JournalScreen}
        options={{ title: 'Journal' }}
      />
      <Stack.Screen 
        name="BreathingExercise" 
        component={BreathingExerciseScreen}
        options={{ title: 'Breathing Exercise' }}
      />
      <Stack.Screen 
        name="Resources" 
        component={ResourcesScreen}
        options={{ title: 'Resources' }}
      />
    </>
  );

  return (
    <SafeAreaProvider>
      <StatusBar barStyle="dark-content" />
      <NavigationContainer children={
        <Stack.Navigator
          initialRouteName="Home"
          screenOptions={{
            headerStyle: {
              backgroundColor: '#f8f9fa',
            },
            headerTintColor: '#2c3e50',
            headerTitleStyle: {
              fontWeight: 'bold',
            },
          }}
          children={screens}
        />
      } />
    </SafeAreaProvider>
  );
};

export default App; 