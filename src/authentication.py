"""
Authentication Module

This module combines liveness detection, deepfake detection, and face embedding
matching to provide comprehensive authentication with optional multi-factor
authentication support.
"""

import logging
import time
import random
import string
import smtplib
from typing import Optional, Dict, List, Tuple, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import hashlib
import json
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OTPManager:
    """
    One-Time Password (OTP) manager for multi-factor authentication.
    """
    
    def __init__(self, 
                 otp_length: int = 6,
                 otp_expiry_minutes: int = 5,
                 smtp_server: str = "smtp.gmail.com",
                 smtp_port: int = 587):
        """
        Initialize OTP manager.
        
        Args:
            otp_length: Length of generated OTP
            otp_expiry_minutes: OTP expiry time in minutes
            smtp_server: SMTP server for email
            smtp_port: SMTP port
        """
        self.otp_length = otp_length
        self.otp_expiry_minutes = otp_expiry_minutes
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        
        # In-memory OTP storage (in production, use Redis or database)
        self.active_otps = {}
        
        logger.info(f"OTPManager initialized with {otp_length}-digit OTPs")
    
    def generate_otp(self, user_id: str) -> str:
        """
        Generate OTP for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Generated OTP string
        """
        try:
            # Generate random OTP
            otp = ''.join(random.choices(string.digits, k=self.otp_length))
            
            # Store with expiry time
            expiry_time = datetime.now() + timedelta(minutes=self.otp_expiry_minutes)
            self.active_otps[user_id] = {
                'otp': otp,
                'expiry': expiry_time,
                'attempts': 0
            }
            
            logger.info(f"Generated OTP for user {user_id}")
            return otp
            
        except Exception as e:
            logger.error(f"OTP generation error: {str(e)}")
            return ""
    
    def verify_otp(self, user_id: str, provided_otp: str) -> bool:
        """
        Verify provided OTP.
        
        Args:
            user_id: User identifier
            provided_otp: OTP provided by user
            
        Returns:
            True if OTP is valid, False otherwise
        """
        try:
            if user_id not in self.active_otps:
                logger.warning(f"No active OTP for user {user_id}")
                return False
            
            otp_data = self.active_otps[user_id]
            
            # Check expiry
            if datetime.now() > otp_data['expiry']:
                logger.warning(f"Expired OTP for user {user_id}")
                del self.active_otps[user_id]
                return False
            
            # Check attempts limit
            if otp_data['attempts'] >= 3:
                logger.warning(f"Too many OTP attempts for user {user_id}")
                del self.active_otps[user_id]
                return False
            
            # Verify OTP
            if provided_otp == otp_data['otp']:
                logger.info(f"OTP verified successfully for user {user_id}")
                del self.active_otps[user_id]
                return True
            else:
                otp_data['attempts'] += 1
                logger.warning(f"Invalid OTP for user {user_id} (attempt {otp_data['attempts']})")
                return False
                
        except Exception as e:
            logger.error(f"OTP verification error: {str(e)}")
            return False
    
    def send_otp_email(self, 
                       user_email: str, 
                       otp: str,
                       sender_email: str,
                       sender_password: str,
                       user_name: str = "User") -> bool:
        """
        Send OTP via email.
        
        Args:
            user_email: Recipient email address
            otp: OTP to send
            sender_email: Sender email address
            sender_password: Sender email password
            user_name: User's name for personalization
            
        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = user_email
            msg['Subject'] = "Face Recognition System - OTP Verification"
            
            # Email body
            body = f"""
            Dear {user_name},
            
            Your OTP for face recognition system authentication is: {otp}
            
            This OTP is valid for {self.otp_expiry_minutes} minutes.
            Please do not share this code with anyone.
            
            If you did not request this authentication, please contact support immediately.
            
            Best regards,
            Face Recognition System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"OTP email sent to {user_email}")
            return True
            
        except Exception as e:
            logger.error(f"OTP email sending error: {str(e)}")
            return False
    
    def cleanup_expired_otps(self) -> int:
        """
        Clean up expired OTPs.
        
        Returns:
            Number of expired OTPs removed
        """
        try:
            current_time = datetime.now()
            expired_users = []
            
            for user_id, otp_data in self.active_otps.items():
                if current_time > otp_data['expiry']:
                    expired_users.append(user_id)
            
            for user_id in expired_users:
                del self.active_otps[user_id]
            
            if expired_users:
                logger.info(f"Cleaned up {len(expired_users)} expired OTPs")
            
            return len(expired_users)
            
        except Exception as e:
            logger.error(f"OTP cleanup error: {str(e)}")
            return 0


class AuthenticationEngine:
    """
    Main authentication engine that combines all security checks.
    """
    
    def __init__(self,
                 face_capture,
                 embedding_extractor,
                 liveness_detector,
                 deepfake_detector,
                 database_manager,
                 config: Dict[str, Any]):
        """
        Initialize authentication engine.
        
        Args:
            face_capture: Face capture instance
            embedding_extractor: Embedding extractor instance
            liveness_detector: Liveness detector instance
            deepfake_detector: Deepfake detector instance
            database_manager: Database manager instance
            config: Configuration dictionary
        """
        self.face_capture = face_capture
        self.embedding_extractor = embedding_extractor
        self.liveness_detector = liveness_detector
        self.deepfake_detector = deepfake_detector
        self.database_manager = database_manager
        self.config = config
        
        # Authentication thresholds
        self.face_similarity_threshold = config.get('face_similarity_threshold', 0.7)
        self.liveness_threshold = config.get('liveness_threshold', 0.5)
        self.deepfake_threshold = config.get('deepfake_threshold', 0.5)
        self.overall_confidence_threshold = config.get('overall_confidence_threshold', 0.6)
        
        # Multi-factor authentication
        self.enable_mfa = config.get('enable_mfa', False)
        self.otp_manager = OTPManager() if self.enable_mfa else None
        
        # Rate limiting
        self.max_attempts_per_hour = config.get('max_attempts_per_hour', 5)
        self.attempt_tracking = {}
        
        logger.info("AuthenticationEngine initialized")
    
    def register_user(self, 
                     user_id: str,
                     name: str,
                     email: Optional[str] = None,
                     phone: Optional[str] = None,
                     capture_duration: int = 5,
                     min_quality_score: float = 0.7) -> Dict[str, Any]:
        """
        Register a new user with face capture and embedding extraction.
        
        Args:
            user_id: Unique user identifier
            name: User's name
            email: User's email address
            phone: User's phone number
            capture_duration: Duration for face capture in seconds
            min_quality_score: Minimum quality score for accepted faces
            
        Returns:
            Registration result dictionary
        """
        try:
            logger.info(f"Starting user registration for: {user_id}")
            
            # Add user to database
            user_added = self.database_manager.add_user(
                user_id=user_id,
                name=name,
                email=email,
                phone=phone,
                metadata={'registration_time': datetime.now().isoformat()}
            )
            
            if not user_added:
                return {
                    'success': False,
                    'message': 'User already exists or database error',
                    'user_id': user_id
                }
            
            # Capture faces
            captured_faces = self.face_capture.capture_from_webcam(duration=capture_duration)
            
            if not captured_faces:
                return {
                    'success': False,
                    'message': 'No faces captured during registration',
                    'user_id': user_id
                }
            
            # Process captured faces
            valid_embeddings = []
            quality_scores = []
            
            for i, face in enumerate(captured_faces):
                # Check face quality
                quality_metrics = self.face_capture.validate_face_quality(face)
                overall_quality = quality_metrics.get('overall_quality', 0.0)
                
                if overall_quality >= min_quality_score:
                    # Extract embedding
                    embedding = self.embedding_extractor.extract_embedding(face)
                    
                    if embedding is not None:
                        valid_embeddings.append(embedding)
                        quality_scores.append(overall_quality)
                        logger.info(f"Valid embedding {i+1} extracted (quality: {overall_quality:.3f})")
            
            if not valid_embeddings:
                return {
                    'success': False,
                    'message': f'No high-quality faces found (minimum quality: {min_quality_score})',
                    'user_id': user_id
                }
            
            # Select best embedding (highest quality)
            best_idx = np.argmax(quality_scores)
            best_embedding = valid_embeddings[best_idx]
            best_quality = quality_scores[best_idx]
            
            # Add embedding to database
            embedding_id = self.database_manager.add_embedding(
                user_id=user_id,
                embedding=best_embedding,
                quality_score=best_quality
            )
            
            if embedding_id is None:
                return {
                    'success': False,
                    'message': 'Failed to store face embedding',
                    'user_id': user_id
                }
            
            logger.info(f"User {user_id} registered successfully with {len(valid_embeddings)} face samples")
            
            return {
                'success': True,
                'message': 'User registered successfully',
                'user_id': user_id,
                'embedding_id': embedding_id,
                'quality_score': best_quality,
                'total_faces_captured': len(captured_faces),
                'valid_faces_processed': len(valid_embeddings)
            }
            
        except Exception as e:
            logger.error(f"User registration error: {str(e)}")
            return {
                'success': False,
                'message': f'Registration failed: {str(e)}',
                'user_id': user_id
            }
    
    def authenticate_user(self, 
                         capture_duration: int = 3,
                         ip_address: str = "unknown") -> Dict[str, Any]:
        """
        Authenticate user using comprehensive face recognition pipeline.
        
        Args:
            capture_duration: Duration for face capture in seconds
            ip_address: Client IP address for logging
            
        Returns:
            Authentication result dictionary
        """
        try:
            start_time = time.time()
            logger.info("Starting user authentication")
            
            # Check rate limiting
            if not self._check_rate_limit(ip_address):
                return {
                    'success': False,
                    'message': 'Too many authentication attempts. Please try again later.',
                    'confidence': 0.0
                }
            
            # Capture face
            captured_faces = self.face_capture.capture_from_webcam(duration=capture_duration)
            
            if not captured_faces:
                self._log_authentication(None, False, 0.0, 0.0, 0.0, ip_address)
                return {
                    'success': False,
                    'message': 'No face detected during authentication',
                    'confidence': 0.0
                }
            
            # Use the best quality face
            best_face = None
            best_quality = 0.0
            
            for face in captured_faces:
                quality_metrics = self.face_capture.validate_face_quality(face)
                quality = quality_metrics.get('overall_quality', 0.0)
                
                if quality > best_quality:
                    best_quality = quality
                    best_face = face
            
            if best_face is None or best_quality < 0.3:
                self._log_authentication(None, False, 0.0, 0.0, 0.0, ip_address)
                return {
                    'success': False,
                    'message': 'Face quality too low for authentication',
                    'confidence': 0.0
                }
            
            # Step 1: Liveness Detection
            liveness_result = self.liveness_detector.comprehensive_liveness_check(best_face)
            liveness_score = liveness_result.get('combined_score', 0.0)
            
            if liveness_score < self.liveness_threshold:
                self._log_authentication(None, False, 0.0, liveness_score, 0.0, ip_address)
                return {
                    'success': False,
                    'message': 'Liveness check failed - potential spoofing detected',
                    'confidence': liveness_score,
                    'liveness_score': liveness_score
                }
            
            # Step 2: Deepfake Detection
            deepfake_result = self.deepfake_detector.comprehensive_deepfake_analysis(best_face)
            real_score = 1.0 - deepfake_result.get('combined_fake_score', 0.0)
            
            if real_score < self.deepfake_threshold:
                self._log_authentication(None, False, 0.0, liveness_score, real_score, ip_address)
                return {
                    'success': False,
                    'message': 'Deepfake detected - authentication denied',
                    'confidence': real_score,
                    'liveness_score': liveness_score,
                    'deepfake_score': real_score
                }
            
            # Step 3: Face Recognition
            embedding = self.embedding_extractor.extract_embedding(best_face)
            
            if embedding is None:
                self._log_authentication(None, False, 0.0, liveness_score, real_score, ip_address)
                return {
                    'success': False,
                    'message': 'Failed to extract face features',
                    'confidence': 0.0
                }
            
            # Find matching user
            auth_result = self.database_manager.authenticate_user(
                embedding, threshold=self.face_similarity_threshold
            )
            
            if auth_result is None:
                self._log_authentication(None, False, 0.0, liveness_score, real_score, ip_address)
                return {
                    'success': False,
                    'message': 'No matching user found',
                    'confidence': 0.0,
                    'liveness_score': liveness_score,
                    'deepfake_score': real_score
                }
            
            # Calculate overall confidence
            face_similarity = auth_result['similarity']
            overall_confidence = self._calculate_overall_confidence(
                face_similarity, liveness_score, real_score
            )
            
            if overall_confidence < self.overall_confidence_threshold:
                self._log_authentication(
                    auth_result['user_id'], False, overall_confidence, 
                    liveness_score, real_score, ip_address
                )
                return {
                    'success': False,
                    'message': 'Authentication confidence too low',
                    'confidence': overall_confidence,
                    'user_id': auth_result['user_id'],
                    'face_similarity': face_similarity,
                    'liveness_score': liveness_score,
                    'deepfake_score': real_score
                }
            
            # Multi-factor authentication if enabled
            if self.enable_mfa and auth_result.get('email'):
                otp = self.otp_manager.generate_otp(auth_result['user_id'])
                
                # In a real implementation, send OTP via email/SMS
                logger.info(f"MFA required for user {auth_result['user_id']}, OTP: {otp}")
                
                return {
                    'success': False,
                    'message': 'Multi-factor authentication required',
                    'mfa_required': True,
                    'user_id': auth_result['user_id'],
                    'confidence': overall_confidence,
                    'face_similarity': face_similarity,
                    'liveness_score': liveness_score,
                    'deepfake_score': real_score
                }
            
            # Successful authentication
            processing_time = time.time() - start_time
            
            self._log_authentication(
                auth_result['user_id'], True, overall_confidence,
                liveness_score, real_score, ip_address
            )
            
            logger.info(f"User {auth_result['user_id']} authenticated successfully in {processing_time:.2f}s")
            
            return {
                'success': True,
                'message': 'Authentication successful',
                'user_id': auth_result['user_id'],
                'name': auth_result['name'],
                'confidence': overall_confidence,
                'face_similarity': face_similarity,
                'liveness_score': liveness_score,
                'deepfake_score': real_score,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return {
                'success': False,
                'message': f'Authentication failed: {str(e)}',
                'confidence': 0.0
            }
    
    def verify_mfa(self, user_id: str, otp: str) -> Dict[str, Any]:
        """
        Verify multi-factor authentication OTP.
        
        Args:
            user_id: User identifier
            otp: One-time password
            
        Returns:
            MFA verification result
        """
        try:
            if not self.enable_mfa or not self.otp_manager:
                return {
                    'success': False,
                    'message': 'Multi-factor authentication not enabled'
                }
            
            if self.otp_manager.verify_otp(user_id, otp):
                logger.info(f"MFA verification successful for user {user_id}")
                return {
                    'success': True,
                    'message': 'Multi-factor authentication successful',
                    'user_id': user_id
                }
            else:
                logger.warning(f"MFA verification failed for user {user_id}")
                return {
                    'success': False,
                    'message': 'Invalid or expired OTP'
                }
                
        except Exception as e:
            logger.error(f"MFA verification error: {str(e)}")
            return {
                'success': False,
                'message': f'MFA verification failed: {str(e)}'
            }
    
    def _calculate_overall_confidence(self, 
                                    face_similarity: float,
                                    liveness_score: float,
                                    deepfake_score: float) -> float:
        """
        Calculate overall authentication confidence score.
        
        Args:
            face_similarity: Face matching similarity score
            liveness_score: Liveness detection score
            deepfake_score: Deepfake detection score (higher = more real)
            
        Returns:
            Overall confidence score (0-1)
        """
        try:
            # Weighted combination of scores
            weights = self.config.get('confidence_weights', {
                'face_similarity': 0.5,
                'liveness': 0.3,
                'deepfake': 0.2
            })
            
            overall_confidence = (
                face_similarity * weights['face_similarity'] +
                liveness_score * weights['liveness'] +
                deepfake_score * weights['deepfake']
            )
            
            return min(1.0, max(0.0, overall_confidence))
            
        except Exception as e:
            logger.error(f"Confidence calculation error: {str(e)}")
            return 0.0
    
    def _check_rate_limit(self, ip_address: str) -> bool:
        """
        Check if IP address has exceeded rate limit.
        
        Args:
            ip_address: Client IP address
            
        Returns:
            True if within rate limit, False otherwise
        """
        try:
            current_time = datetime.now()
            one_hour_ago = current_time - timedelta(hours=1)
            
            # Clean old attempts
            if ip_address in self.attempt_tracking:
                self.attempt_tracking[ip_address] = [
                    attempt_time for attempt_time in self.attempt_tracking[ip_address]
                    if attempt_time > one_hour_ago
                ]
            else:
                self.attempt_tracking[ip_address] = []
            
            # Check current attempts
            if len(self.attempt_tracking[ip_address]) >= self.max_attempts_per_hour:
                logger.warning(f"Rate limit exceeded for IP: {ip_address}")
                return False
            
            # Record this attempt
            self.attempt_tracking[ip_address].append(current_time)
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check error: {str(e)}")
            return True  # Allow on error
    
    def _log_authentication(self,
                          user_id: Optional[str],
                          success: bool,
                          confidence: float,
                          liveness_score: float,
                          deepfake_score: float,
                          ip_address: str) -> None:
        """
        Log authentication attempt.
        
        Args:
            user_id: User ID (None for failed attempts)
            success: Whether authentication was successful
            confidence: Overall confidence score
            liveness_score: Liveness detection score
            deepfake_score: Deepfake detection score
            ip_address: Client IP address
        """
        try:
            self.database_manager.log_authentication(
                user_id=user_id,
                success=success,
                confidence_score=confidence,
                liveness_score=liveness_score,
                deepfake_score=deepfake_score,
                ip_address=ip_address,
                metadata={'timestamp': datetime.now().isoformat()}
            )
        except Exception as e:
            logger.error(f"Authentication logging error: {str(e)}")
    
    def get_authentication_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get authentication statistics.
        
        Args:
            user_id: Optional user ID to filter stats
            
        Returns:
            Authentication statistics
        """
        try:
            return self.database_manager.get_statistics()
        except Exception as e:
            logger.error(f"Authentication stats error: {str(e)}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Mock configuration
    config = {
        'face_similarity_threshold': 0.7,
        'liveness_threshold': 0.5,
        'deepfake_threshold': 0.5,
        'overall_confidence_threshold': 0.6,
        'enable_mfa': False,
        'max_attempts_per_hour': 5,
        'confidence_weights': {
            'face_similarity': 0.5,
            'liveness': 0.3,
            'deepfake': 0.2
        }
    }
    
    # Test OTP manager
    otp_manager = OTPManager()
    
    # Generate and verify OTP
    test_user_id = "test_user_001"
    otp = otp_manager.generate_otp(test_user_id)
    print(f"Generated OTP: {otp}")
    
    # Test verification
    verification_result = otp_manager.verify_otp(test_user_id, otp)
    print(f"OTP verification result: {verification_result}")
    
    # Test invalid OTP
    invalid_result = otp_manager.verify_otp(test_user_id, "wrong_otp")
    print(f"Invalid OTP result: {invalid_result}")
    
    print("Authentication module test completed")
