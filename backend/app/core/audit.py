"""
Audit Logging Module for HIPAA Compliance
Logs all system actions for audit trails and compliance
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from functools import wraps
from fastapi import Request

logger = logging.getLogger(__name__)

# Audit log file path
AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", "logs/audit.log")

class AuditLogger:
    """
    Centralized audit logging for HIPAA compliance.
    Logs all user actions, data access, and system events.
    """
    
    def __init__(self):
        self.enabled = os.getenv("ENABLE_AUDIT_LOGGING", "true").lower() == "true"
        
        # Ensure log directory exists
        if self.enabled:
            log_dir = os.path.dirname(AUDIT_LOG_PATH)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
    
    def log_action(
        self,
        user: str,
        action: str,
        resource: str,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        success: bool = True
    ) -> None:
        """
        Log an audit action.
        
        Args:
            user: Username or user identifier
            action: Action performed (e.g., "LOGIN", "EXTRACT", "VERIFY")
            resource: Resource accessed (e.g., "/api/v1/extract", "extraction_123")
            details: Additional details about the action
            ip_address: IP address of the user
            success: Whether the action was successful
        """
        if not self.enabled:
            return
        
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user": user,
            "action": action,
            "resource": resource,
            "details": details or {},
            "ip_address": ip_address,
            "success": success
        }
        
        # Log to file
        try:
            with open(AUDIT_LOG_PATH, "a") as f:
                f.write(json.dumps(audit_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
        
        # Also log to standard logger
        log_level = logging.INFO if success else logging.WARNING
        logger.log(
            log_level,
            f"AUDIT: {user} {action} {resource} - Success: {success}"
        )
    
    def log_extraction(
        self,
        user: str,
        extraction_id: int,
        text_length: int,
        entity_count: int,
        phi_detected: bool,
        ip_address: Optional[str] = None
    ) -> None:
        """Log an extraction action"""
        self.log_action(
            user=user,
            action="EXTRACT",
            resource=f"extraction_{extraction_id}",
            details={
                "text_length": text_length,
                "entity_count": entity_count,
                "phi_detected": phi_detected
            },
            ip_address=ip_address,
            success=True
        )
    
    def log_verification(
        self,
        user: str,
        extraction_id: int,
        changes_made: int,
        ip_address: Optional[str] = None
    ) -> None:
        """Log a verification action"""
        self.log_action(
            user=user,
            action="VERIFY",
            resource=f"extraction_{extraction_id}",
            details={"changes_made": changes_made},
            ip_address=ip_address,
            success=True
        )
    
    def log_login(
        self,
        user: str,
        success: bool,
        ip_address: Optional[str] = None
    ) -> None:
        """Log a login attempt"""
        self.log_action(
            user=user,
            action="LOGIN",
            resource="/api/v1/auth/login",
            ip_address=ip_address,
            success=success
        )
    
    def log_data_access(
        self,
        user: str,
        resource: str,
        access_type: str = "READ",
        ip_address: Optional[str] = None
    ) -> None:
        """Log data access"""
        self.log_action(
            user=user,
            action=f"DATA_ACCESS_{access_type}",
            resource=resource,
            ip_address=ip_address,
            success=True
        )


# Global singleton instance
audit_logger = AuditLogger()

def audit_log(action: str, resource: str):
    """
    Decorator for automatic audit logging of API endpoints.
    
    Args:
        action: Action being performed
        resource: Resource being accessed
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request if available
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            # Get user from request if authenticated
            user = "anonymous"
            ip_address = None
            
            if request:
                ip_address = request.client.host if request.client else None
                # Try to get user from request state (set by auth dependency)
                if hasattr(request.state, "user"):
                    user = request.state.user.username
            
            # Execute function
            success = True
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                # Log the action
                audit_logger.log_action(
                    user=user,
                    action=action,
                    resource=resource,
                    ip_address=ip_address,
                    success=success
                )
        
        return wrapper
    return decorator
