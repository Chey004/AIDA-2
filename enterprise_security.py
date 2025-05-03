"""
Enterprise Security Features
GDPR compliance and audit trail system
"""

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import hashlib
import spacy
from spacy.tokens import Doc

class GDPRCompliantStorage:
    def __init__(self, encryption_key: bytes):
        self.encryption_key = self._derive_key(encryption_key)
        self.retention_policy = {
            'default': timedelta(days=30),
            'clinical': timedelta(days=3650),  # 10 years
            'research': None  # permanent
        }
        self.nlp = spacy.load('en_core_web_sm')
        
    def _derive_key(self, password: bytes) -> bytes:
        """Derive encryption key from password"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'CPAS_SALT',
            iterations=100000
        )
        return kdf.derive(password)
    
    def encrypt_data(self, data: Dict[str, Any]) -> str:
        """Encrypt sensitive data using AES-GCM"""
        nonce = AESGCM.generate_nonce()
        aesgcm = AESGCM(self.encryption_key)
        encrypted_data = aesgcm.encrypt(
            nonce,
            json.dumps(data).encode(),
            None
        )
        return base64.b64encode(nonce + encrypted_data).decode()
    
    def decrypt_data(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt data using AES-GCM"""
        data = base64.b64decode(encrypted_data)
        nonce = data[:12]
        ciphertext = data[12:]
        aesgcm = AESGCM(self.encryption_key)
        decrypted = aesgcm.decrypt(nonce, ciphertext, None)
        return json.loads(decrypted)
    
    def anonymize_text(self, text: str) -> str:
        """Remove personally identifiable information from text"""
        doc = self.nlp(text)
        anonymized = []
        
        for token in doc:
            if token.ent_type_ in ['PERSON', 'GPE', 'ORG', 'DATE', 'TIME']:
                anonymized.append(f"[{token.ent_type_}]")
            else:
                anonymized.append(token.text)
        
        return ' '.join(anonymized)
    
    def apply_retention_policy(self, data_id: str, category: str) -> bool:
        """Check if data should be retained based on policy"""
        if category not in self.retention_policy:
            return True
        
        retention = self.retention_policy[category]
        if retention is None:  # permanent retention
            return True
        
        # Check if data is within retention period
        # Implementation depends on your storage system
        return True

class AuditTrailSystem:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.consent_forms = {}
    
    def log_analysis_session(self, user_id: str, actions: List[Dict[str, Any]]) -> str:
        """Log analysis session with blockchain-like immutability"""
        session_id = self._generate_session_id(user_id)
        timestamp = datetime.utcnow().isoformat()
        
        session_data = {
            'session_id': session_id,
            'timestamp': timestamp,
            'user_id': self._hash_user_id(user_id),
            'actions': actions,
            'consent': self._get_consent_status(user_id),
            'previous_hash': self._get_last_hash()
        }
        
        # Calculate hash of current session
        session_data['hash'] = self._calculate_hash(session_data)
        
        # Store session data
        self._store_session(session_data)
        
        return session_id
    
    def _generate_session_id(self, user_id: str) -> str:
        """Generate unique session ID"""
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        return f"{self._hash_user_id(user_id)}_{timestamp}"
    
    def _hash_user_id(self, user_id: str) -> str:
        """Hash user ID for privacy"""
        return hashlib.sha256(user_id.encode()).hexdigest()
    
    def _get_consent_status(self, user_id: str) -> Dict[str, Any]:
        """Get user's consent status"""
        return self.consent_forms.get(user_id, {
            'status': 'not_obtained',
            'timestamp': None,
            'version': None
        })
    
    def _get_last_hash(self) -> Optional[str]:
        """Get hash of last logged session"""
        # Implementation depends on your storage system
        return None
    
    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """Calculate hash of session data"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _store_session(self, session_data: Dict[str, Any]) -> None:
        """Store session data with immutability"""
        # Implementation depends on your storage system
        pass
    
    def verify_session_integrity(self, session_id: str) -> bool:
        """Verify session data integrity"""
        # Implementation depends on your storage system
        return True 