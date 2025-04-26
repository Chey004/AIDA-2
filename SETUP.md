# Mental Health Buddy App - Development Setup Guide

## Prerequisites

1. **Node.js and npm**
   - Download and install from https://nodejs.org/
   - Recommended version: LTS (Long Term Support)

2. **Android Studio** (for Android development)
   - Download and install from https://developer.android.com/studio
   - During installation, make sure to select:
     - Android SDK
     - Android SDK Platform
     - Android Virtual Device (AVD)

3. **Environment Variables** (for Android development)
   - Run the `setup-env.ps1` script in PowerShell
   - Add the following environment variables to your system:
     - ANDROID_HOME: %LOCALAPPDATA%\Android\Sdk
     - Add to Path:
       - %LOCALAPPDATA%\Android\Sdk\platform-tools
       - %LOCALAPPDATA%\Android\Sdk\tools
       - %LOCALAPPDATA%\Android\Sdk\tools\bin

## Project Setup

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Initialize React Native Project**
   ```bash
   npx react-native init MentalHealthBuddy --directory .
   ```

## Running the App

### Android
1. **Start Metro Bundler**
   ```bash
   npx react-native start
   ```

2. **Run the App**
   ```bash
   npx react-native run-android
   ```

### Web
1. **Start Web Development Server**
   ```bash
   npm run web
   ```

2. **Build for Production**
   ```bash
   npm run build:web
   ```

## Troubleshooting

1. **Android SDK not found**
   - Make sure Android Studio is installed
   - Verify environment variables are set correctly
   - Restart your terminal after setting environment variables

2. **Metro Bundler issues**
   - Clear Metro cache: `npx react-native start --reset-cache`
   - Make sure no other Metro instances are running

3. **Build errors**
   - Clean the project: `cd android && ./gradlew clean`
   - Rebuild: `npx react-native run-android`

4. **Web build issues**
   - Make sure all web dependencies are installed
   - Check for compatibility issues between React and React DOM versions
   - Clear webpack cache: `npm run web -- --reset-cache`

## Development Tools

- **VS Code Extensions**
  - React Native Tools
  - ESLint
  - Prettier
  - TypeScript and JavaScript Language Features

- **Android Studio Plugins**
  - React Native
  - Kotlin
  - Android SDK 