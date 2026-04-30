"""
Prometheus Metrics Service
Exposes application metrics for monitoring and alerting
"""

import logging
import os
import time
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
from functools import wraps

logger = logging.getLogger(__name__)

# Prometheus configuration
ENABLE_PROMETHEUS = os.getenv("ENABLE_PROMETHEUS", "false").lower() == "true"
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", 9090))

class PrometheusMetrics:
    """
    Prometheus metrics collector for application monitoring.
    """
    
    def __init__(self):
        self.enabled = ENABLE_PROMETHEUS
        self._initialize_metrics()
        
        if self.enabled:
            try:
                start_http_server(PROMETHEUS_PORT)
                logger.info(f"Prometheus metrics server started on port {PROMETHEUS_PORT}")
            except Exception as e:
                logger.error(f"Failed to start Prometheus metrics server: {e}")
                self.enabled = False
    
    def _initialize_metrics(self):
        """Initialize Prometheus metrics"""
        if not self.enabled:
            return
        
        # Counter metrics
        self.extraction_requests_total = Counter(
            'extraction_requests_total',
            'Total number of extraction requests',
            ['model_version', 'status']
        )
        
        self.extraction_entities_total = Counter(
            'extraction_entities_total',
            'Total number of entities extracted',
            ['entity_type']
        )
        
        self.phi_detections_total = Counter(
            'phi_detections_total',
            'Total number of PHI detections',
            ['phi_type']
        )
        
        self.auth_requests_total = Counter(
            'auth_requests_total',
            'Total number of authentication requests',
            ['action', 'status']
        )
        
        self.cache_hits_total = Counter(
            'cache_hits_total',
            'Total number of cache hits'
        )
        
        self.cache_misses_total = Counter(
            'cache_misses_total',
            'Total number of cache misses'
        )
        
        # Histogram metrics
        self.extraction_duration_seconds = Histogram(
            'extraction_duration_seconds',
            'Extraction processing duration in seconds',
            ['model_version']
        )
        
        self.api_request_duration_seconds = Histogram(
            'api_request_duration_seconds',
            'API request duration in seconds',
            ['endpoint', 'method']
        )
        
        # Gauge metrics
        self.active_extractions = Gauge(
            'active_extractions',
            'Number of currently active extractions'
        )
        
        self.cache_size = Gauge(
            'cache_size',
            'Current cache size'
        )
        
        # Info metrics
        self.app_info = Info(
            'application_info',
            'Application information'
        )
        self.app_info.info({
            'version': '1.0.0',
            'name': 'MedTex Clinical NER'
        })
    
    def record_extraction_request(self, model_version: str, status: str):
        """Record an extraction request"""
        if self.enabled:
            self.extraction_requests_total.labels(model_version=model_version, status=status).inc()
    
    def record_entity_extraction(self, entity_type: str):
        """Record an entity extraction"""
        if self.enabled:
            self.extraction_entities_total.labels(entity_type=entity_type).inc()
    
    def record_phi_detection(self, phi_type: str):
        """Record a PHI detection"""
        if self.enabled:
            self.phi_detections_total.labels(phi_type=phi_type).inc()
    
    def record_auth_request(self, action: str, status: str):
        """Record an authentication request"""
        if self.enabled:
            self.auth_requests_total.labels(action=action, status=status).inc()
    
    def record_cache_hit(self):
        """Record a cache hit"""
        if self.enabled:
            self.cache_hits_total.inc()
    
    def record_cache_miss(self):
        """Record a cache miss"""
        if self.enabled:
            self.cache_misses_total.inc()
    
    def time_extraction(self, model_version: str):
        """Decorator to time extraction operations"""
        if not self.enabled:
            return lambda f: f
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.extraction_duration_seconds.labels(model_version=model_version).observe(duration)
            return wrapper
        return decorator
    
    def time_api_request(self, endpoint: str, method: str):
        """Decorator to time API requests"""
        if not self.enabled:
            return lambda f: f
        
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.api_request_duration_seconds.labels(endpoint=endpoint, method=method).observe(duration)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.api_request_duration_seconds.labels(endpoint=endpoint, method=method).observe(duration)
            
            # Return appropriate wrapper based on function type
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        return decorator
    
    def set_active_extractions(self, count: int):
        """Set the number of active extractions"""
        if self.enabled:
            self.active_extractions.set(count)
    
    def set_cache_size(self, size: int):
        """Set the current cache size"""
        if self.enabled:
            self.cache_size.set(size)


# Global singleton instance
prometheus_metrics = PrometheusMetrics()
