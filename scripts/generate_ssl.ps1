# Configuration
$Domain = "yourdomain.com"
$Email = "admin@yourdomain.com"
$SSLDir = "nginx\ssl"

# Create SSL directory if it doesn't exist
New-Item -ItemType Directory -Force -Path $SSLDir | Out-Null

# Generate self-signed certificate (for development)
Write-Host "Generating SSL certificates..."
$cert = New-SelfSignedCertificate -DnsName $Domain -CertStoreLocation "cert:\LocalMachine\My" -NotAfter (Get-Date).AddYears(1)
$certPath = "cert:\LocalMachine\My\$($cert.Thumbprint)"

# Export certificate and private key
Export-Certificate -Cert $certPath -FilePath "$SSLDir\cert.pem" -Type CERT
$pwd = ConvertTo-SecureString -String "YourSecurePassword" -Force -AsPlainText
Export-PfxCertificate -Cert $certPath -FilePath "$SSLDir\cert.pfx" -Password $pwd

# Convert PFX to PEM format for private key
openssl pkcs12 -in "$SSLDir\cert.pfx" -out "$SSLDir\key.pem" -nodes -password pass:YourSecurePassword

# Set proper permissions
$acl = Get-Acl $SSLDir
$acl.SetAccessRuleProtection($true, $false)
$rule = New-Object System.Security.AccessControl.FileSystemAccessRule("Administrators","FullControl","Allow")
$acl.SetAccessRule($rule)
Set-Acl $SSLDir $acl

Write-Host "SSL certificates generated in $SSLDir" -ForegroundColor Green 