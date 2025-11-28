"""
API Module

FastAPI-based REST API for face recognition system providing endpoints
for user registration, authentication, and system management.
"""

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import uvicorn
import logging
import time
import json
import numpy as np
import cv2
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, EmailStr
from PIL import Image
import io
import asyncio
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response validation
class UserRegistrationRequest(BaseModel):
    user_id: str
    name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    capture_duration: Optional[int] = 5
    min_quality_score: Optional[float] = 0.7

class AuthenticationRequest(BaseModel):
    capture_duration: Optional[int] = 3

class MFARequest(BaseModel):
    user_id: str
    otp: str

class UserResponse(BaseModel):
    user_id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    created_at: str
    is_active: bool

class AuthenticationResponse(BaseModel):
    success: bool
    message: str
    user_id: Optional[str] = None
    name: Optional[str] = None
    confidence: float
    face_similarity: Optional[float] = None
    liveness_score: Optional[float] = None
    deepfake_score: Optional[float] = None
    processing_time: Optional[float] = None
    mfa_required: Optional[bool] = False

class RegistrationResponse(BaseModel):
    success: bool
    message: str
    user_id: str
    embedding_id: Optional[str] = None
    quality_score: Optional[float] = None
    total_faces_captured: Optional[int] = None
    valid_faces_processed: Optional[int] = None

class SystemStatsResponse(BaseModel):
    total_users: int
    active_users: int
    total_embeddings: int
    avg_embedding_quality: float
    total_authentications: int
    successful_authentications: int
    success_rate_percent: float
    faiss_index_size: int


class FaceRecognitionAPI:
    """
    Main FastAPI application class for face recognition system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FastAPI application.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.app = FastAPI(
            title="Face Recognition System API",
            description="Secure face recognition with anti-spoofing and deepfake detection",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=config.get("allowed_origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Security
        self.security = HTTPBearer(auto_error=False)
        
        # Initialize components (will be set by main.py)
        self.face_capture = None
        self.embedding_extractor = None
        self.liveness_detector = None
        self.deepfake_detector = None
        self.database_manager = None
        self.auth_engine = None
        
        # Setup routes
        self._setup_routes()
        
        logger.info("FastAPI application initialized")
    
    def set_components(self, 
                      face_capture,
                      embedding_extractor,
                      liveness_detector,
                      deepfake_detector,
                      database_manager,
                      auth_engine):
        """
        Set system components after initialization.
        
        Args:
            face_capture: Face capture instance
            embedding_extractor: Embedding extractor instance
            liveness_detector: Liveness detector instance
            deepfake_detector: Deepfake detector instance
            database_manager: Database manager instance
            auth_engine: Authentication engine instance
        """
        self.face_capture = face_capture
        self.embedding_extractor = embedding_extractor
        self.liveness_detector = liveness_detector
        self.deepfake_detector = deepfake_detector
        self.database_manager = database_manager
        self.auth_engine = auth_engine
        
        logger.info("System components set successfully")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint with API information."""
            return {
                "message": "Face Recognition System API",
                "version": "1.0.0",
                "status": "active",
                "timestamp": datetime.now().isoformat(),
                "endpoints": {
                    "registration": "/api/v1/register",
                    "authentication": "/api/v1/authenticate",
                    "mfa_verification": "/api/v1/verify-mfa",
                    "user_info": "/api/v1/users/{user_id}",
                    "system_stats": "/api/v1/stats",
                    "health_check": "/api/v1/health"
                }
            }
        
        @self.app.get("/api/v1/health")
        async def health_check():
            """Health check endpoint."""
            try:
                # Check if all components are initialized
                components_status = {
                    "face_capture": self.face_capture is not None,
                    "embedding_extractor": self.embedding_extractor is not None,
                    "liveness_detector": self.liveness_detector is not None,
                    "deepfake_detector": self.deepfake_detector is not None,
                    "database_manager": self.database_manager is not None,
                    "auth_engine": self.auth_engine is not None
                }
                
                all_healthy = all(components_status.values())
                
                return {
                    "status": "healthy" if all_healthy else "unhealthy",
                    "timestamp": datetime.now().isoformat(),
                    "components": components_status,
                    "uptime": time.time()  # This would be calculated properly in production
                }
                
            except Exception as e:
                logger.error(f"Health check error: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "unhealthy",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                )
        
        @self.app.post("/api/v1/register", response_model=RegistrationResponse)
        async def register_user(request: UserRegistrationRequest, client_request: Request):
            """
            Register a new user with face capture and embedding extraction.
            
            Args:
                request: User registration data
                client_request: FastAPI request object
                
            Returns:
                Registration result
            """
            try:
                if not self.auth_engine:
                    raise HTTPException(status_code=500, detail="Authentication engine not initialized")
                
                client_ip = client_request.client.host
                logger.info(f"User registration request from {client_ip} for user: {request.user_id}")
                
                # Call authentication engine for registration
                result = self.auth_engine.register_user(
                    user_id=request.user_id,
                    name=request.name,
                    email=request.email,
                    phone=request.phone,
                    capture_duration=request.capture_duration,
                    min_quality_score=request.min_quality_score
                )
                
                if result['success']:
                    logger.info(f"User {request.user_id} registered successfully")
                    return RegistrationResponse(**result)
                else:
                    logger.warning(f"User registration failed: {result['message']}")
                    raise HTTPException(status_code=400, detail=result['message'])
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Registration endpoint error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")
        
        @self.app.post("/api/v1/authenticate", response_model=AuthenticationResponse)
        async def authenticate_user(request: AuthenticationRequest, client_request: Request):
            """
            Authenticate user using face recognition pipeline.
            
            Args:
                request: Authentication request data
                client_request: FastAPI request object
                
            Returns:
                Authentication result
            """
            try:
                if not self.auth_engine:
                    raise HTTPException(status_code=500, detail="Authentication engine not initialized")
                
                client_ip = client_request.client.host
                logger.info(f"Authentication request from {client_ip}")
                
                # Call authentication engine
                result = self.auth_engine.authenticate_user(
                    capture_duration=request.capture_duration,
                    ip_address=client_ip
                )
                
                # Return appropriate HTTP status
                if result['success']:
                    logger.info(f"Authentication successful for user: {result.get('user_id', 'unknown')}")
                    return AuthenticationResponse(**result)
                elif result.get('mfa_required', False):
                    logger.info(f"MFA required for user: {result.get('user_id', 'unknown')}")
                    return AuthenticationResponse(**result)
                else:
                    logger.warning(f"Authentication failed: {result['message']}")
                    return AuthenticationResponse(**result)
                    
            except Exception as e:
                logger.error(f"Authentication endpoint error: {str(e)}")
                return AuthenticationResponse(
                    success=False,
                    message=f"Authentication failed: {str(e)}",
                    confidence=0.0
                )
        
        @self.app.post("/api/v1/verify-mfa")
        async def verify_mfa(request: MFARequest):
            """
            Verify multi-factor authentication OTP.
            
            Args:
                request: MFA verification request
                
            Returns:
                MFA verification result
            """
            try:
                if not self.auth_engine:
                    raise HTTPException(status_code=500, detail="Authentication engine not initialized")
                
                logger.info(f"MFA verification request for user: {request.user_id}")
                
                result = self.auth_engine.verify_mfa(request.user_id, request.otp)
                
                if result['success']:
                    logger.info(f"MFA verification successful for user: {request.user_id}")
                    return result
                else:
                    logger.warning(f"MFA verification failed for user: {request.user_id}")
                    raise HTTPException(status_code=400, detail=result['message'])
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"MFA verification endpoint error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"MFA verification failed: {str(e)}")
        
        @self.app.get("/api/v1/users/{user_id}", response_model=UserResponse)
        async def get_user(user_id: str, credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            """
            Get user information by user ID.
            
            Args:
                user_id: User identifier
                credentials: HTTP authorization credentials
                
            Returns:
                User information
            """
            try:
                # In production, validate credentials here
                if not self.database_manager:
                    raise HTTPException(status_code=500, detail="Database manager not initialized")
                
                user_info = self.database_manager.get_user(user_id)
                
                if user_info:
                    return UserResponse(**user_info)
                else:
                    raise HTTPException(status_code=404, detail="User not found")
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Get user endpoint error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to get user: {str(e)}")
        
        @self.app.delete("/api/v1/users/{user_id}")
        async def delete_user(user_id: str, credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            """
            Delete user and all associated data.
            
            Args:
                user_id: User identifier
                credentials: HTTP authorization credentials
                
            Returns:
                Deletion result
            """
            try:
                # In production, validate credentials and check permissions
                if not self.database_manager:
                    raise HTTPException(status_code=500, detail="Database manager not initialized")
                
                success = self.database_manager.delete_user(user_id)
                
                if success:
                    logger.info(f"User {user_id} deleted successfully")
                    return {"message": f"User {user_id} deleted successfully"}
                else:
                    raise HTTPException(status_code=404, detail="User not found or deletion failed")
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Delete user endpoint error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to delete user: {str(e)}")
        
        @self.app.get("/api/v1/stats", response_model=SystemStatsResponse)
        async def get_system_stats(credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            """
            Get system statistics.
            
            Args:
                credentials: HTTP authorization credentials
                
            Returns:
                System statistics
            """
            try:
                # In production, validate credentials here
                if not self.database_manager:
                    raise HTTPException(status_code=500, detail="Database manager not initialized")
                
                stats = self.database_manager.get_statistics()
                
                if stats:
                    return SystemStatsResponse(**stats)
                else:
                    raise HTTPException(status_code=500, detail="Failed to retrieve statistics")
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"System stats endpoint error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")
        
        @self.app.post("/api/v1/upload-image")
        async def upload_image_for_analysis(file: UploadFile = File(...)):
            """
            Upload image for face analysis (liveness, deepfake detection).
            
            Args:
                file: Uploaded image file
                
            Returns:
                Analysis results
            """
            try:
                if not (self.liveness_detector and self.deepfake_detector):
                    raise HTTPException(status_code=500, detail="Detection models not initialized")
                
                # Read and validate image
                contents = await file.read()
                
                try:
                    image = Image.open(io.BytesIO(contents))
                    image_np = np.array(image)
                    
                    if len(image_np.shape) != 3 or image_np.shape[2] != 3:
                        raise HTTPException(status_code=400, detail="Invalid image format. RGB images required.")
                    
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
                
                # Perform analysis
                liveness_result = self.liveness_detector.comprehensive_liveness_check(image_np)
                deepfake_result = self.deepfake_detector.comprehensive_deepfake_analysis(image_np)
                
                return {
                    "filename": file.filename,
                    "image_size": image_np.shape,
                    "liveness_analysis": {
                        "is_live": liveness_result.get('is_live', False),
                        "confidence": liveness_result.get('confidence', 0.0),
                        "combined_score": liveness_result.get('combined_score', 0.0)
                    },
                    "deepfake_analysis": {
                        "is_deepfake": deepfake_result.get('is_deepfake', True),
                        "confidence": deepfake_result.get('confidence', 0.0),
                        "combined_fake_score": deepfake_result.get('combined_fake_score', 1.0)
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Image upload endpoint error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")
        
        @self.app.get("/api/v1/system/benchmark")
        async def benchmark_system(credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            """
            Benchmark system performance.
            
            Args:
                credentials: HTTP authorization credentials
                
            Returns:
                Benchmark results
            """
            try:
                # In production, validate credentials here
                benchmarks = {}
                
                if self.embedding_extractor:
                    benchmarks['embedding_extraction'] = self.embedding_extractor.benchmark_inference_time(10)
                
                if self.liveness_detector:
                    benchmarks['liveness_detection'] = self.liveness_detector.benchmark_inference_time(10)
                
                if self.deepfake_detector:
                    benchmarks['deepfake_detection'] = self.deepfake_detector.benchmark_inference_time(10)
                
                return {
                    "benchmarks": benchmarks,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Benchmark endpoint error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")
        
        # Exception handlers
        @self.app.exception_handler(404)
        async def not_found_handler(request: Request, exc: HTTPException):
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Not Found",
                    "message": "The requested resource was not found",
                    "path": request.url.path,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        @self.app.exception_handler(500)
        async def internal_error_handler(request: Request, exc: HTTPException):
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An internal server error occurred",
                    "path": request.url.path,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        """
        Run the FastAPI application.
        
        Args:
            host: Host address
            port: Port number
            debug: Debug mode
        """
        try:
            logger.info(f"Starting Face Recognition API server on {host}:{port}")
            uvicorn.run(
                self.app,
                host=host,
                port=port,
                reload=debug,
                log_level="info" if not debug else "debug"
            )
        except Exception as e:
            logger.error(f"Failed to start API server: {str(e)}")
            raise


# Utility functions for API
def validate_image_file(file: UploadFile) -> bool:
    """
    Validate uploaded image file.
    
    Args:
        file: Uploaded file
        
    Returns:
        True if valid image file, False otherwise
    """
    try:
        # Check file extension
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return False
        
        # Check MIME type
        allowed_mime_types = {'image/jpeg', 'image/png', 'image/bmp', 'image/tiff'}
        
        if file.content_type not in allowed_mime_types:
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Image validation error: {str(e)}")
        return False


def create_api_response(success: bool, 
                       message: str, 
                       data: Optional[Dict] = None,
                       error_code: Optional[str] = None) -> Dict[str, Any]:
    """
    Create standardized API response.
    
    Args:
        success: Success status
        message: Response message
        data: Optional response data
        error_code: Optional error code
        
    Returns:
        Standardized response dictionary
    """
    response = {
        "success": success,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    
    if data:
        response["data"] = data
    
    if error_code:
        response["error_code"] = error_code
    
    return response


# Example usage and testing
if __name__ == "__main__":
    # Mock configuration for testing
    config = {
        "allowed_origins": ["*"],
        "api_key_required": False
    }
    
    # Create API instance
    api = FaceRecognitionAPI(config)
    
    # In production, components would be initialized by main.py
    print("API routes configured:")
    for route in api.app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            methods = list(route.methods)
            print(f"  {', '.join(methods)} {route.path}")
    
    print("\nTo run the API server, use: python main.py")
