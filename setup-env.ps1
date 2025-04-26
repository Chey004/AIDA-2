# Set Android SDK environment variables
$androidSdkPath = "$env:LOCALAPPDATA\Android\Sdk"
$env:ANDROID_HOME = $androidSdkPath
$env:Path += ";$androidSdkPath\platform-tools"
$env:Path += ";$androidSdkPath\tools"
$env:Path += ";$androidSdkPath\tools\bin"

# Create a temporary file to store the environment variables
$envFile = "$env:USERPROFILE\.react-native-env"
"ANDROID_HOME=$androidSdkPath" | Out-File -FilePath $envFile -Encoding utf8
"Path=$env:Path" | Out-File -FilePath $envFile -Encoding utf8 -Append

Write-Host "Environment variables have been set temporarily."
Write-Host "To make these changes permanent, please add the following to your system environment variables:"
Write-Host "ANDROID_HOME: $androidSdkPath"
Write-Host "Add to Path: $androidSdkPath\platform-tools"
Write-Host "Add to Path: $androidSdkPath\tools"
Write-Host "Add to Path: $androidSdkPath\tools\bin" 