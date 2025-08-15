{
    # Global options for geo-distributed load balancing
    admin off
    auto_https on
    
    # Enable geo-location module (if available)
    # order geoip first
}

# Multi-region configuration
:80, :443 {
    # Geo-based routing using headers or IP detection
    @us_west {
        header CF-IPCountry US
        header CF-Region WEST
    }
    
    @eu {
        header CF-IPCountry GB DE FR ES IT NL
    }
    
    @asia {
        header CF-IPCountry JP CN KR SG IN
    }

    # API routes with regional backends
    @api {
        path /api/* /docs* /openapi.json /health /ws/*
    }
    
    # Route US West traffic
    handle @us_west {
        handle @api {
            reverse_proxy {
                # US West API instances (Phoenix, LA, Seattle)
                to http://agent-core-api-phx.internal:8000
                to http://agent-core-api-lax.internal:8000
                to http://agent-core-api-sea.internal:8000
                
                lb_policy least_conn
                health_uri /api/health
                health_interval 5s
                
                # Fallback to other regions if needed
                fallback http://agent-core-api.internal:8000
            }
        }
        handle {
            reverse_proxy http://agent-core-web-phx.internal:8001
        }
    }
    
    # Route EU traffic
    handle @eu {
        handle @api {
            reverse_proxy {
                # EU API instances (London, Frankfurt, Amsterdam)
                to http://agent-core-api-lhr.internal:8000
                to http://agent-core-api-fra.internal:8000
                to http://agent-core-api-ams.internal:8000
                
                lb_policy least_conn
                health_uri /api/health
                health_interval 5s
                
                fallback http://agent-core-api.internal:8000
            }
        }
        handle {
            reverse_proxy http://agent-core-web-lhr.internal:8001
        }
    }
    
    # Route Asia traffic
    handle @asia {
        handle @api {
            reverse_proxy {
                # Asia API instances (Tokyo, Singapore, Sydney)
                to http://agent-core-api-nrt.internal:8000
                to http://agent-core-api-sin.internal:8000
                to http://agent-core-api-syd.internal:8000
                
                lb_policy least_conn
                health_uri /api/health
                health_interval 5s
                
                fallback http://agent-core-api.internal:8000
            }
        }
        handle {
            reverse_proxy http://agent-core-web-sin.internal:8001
        }
    }
    
    # Default routing (for unmatched regions)
    handle @api {
        reverse_proxy {
            # Use Fly.io's automatic regional routing
            to http://agent-core-api.internal:8000
            
            # Dynamic upstream discovery
            # Fly.io will route to nearest healthy instance
            lb_policy least_conn
            health_uri /api/health
            health_interval 5s
            health_timeout 2s
            
            # Advanced health checks
            health_headers {
                X-Health-Check "true"
            }
            
            # Retry with exponential backoff
            lb_retries 3
            lb_try_duration 10s
            lb_try_interval 250ms
        }
    }
    
    handle {
        reverse_proxy {
            to http://agent-core-web.internal:8001
            lb_policy ip_hash  # Sticky sessions for web
        }
    }

    # Add geo-location headers for debugging
    header {
        X-Served-By "{system.hostname}"
        X-Region "{env.FLY_REGION}"
        X-Request-ID "{http.request.uuid}"
    }

    # Enable compression
    encode gzip zstd

    # Logging with geo info
    log {
        output stdout
        format json
        level INFO
    }
}