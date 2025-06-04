# Configuration Migration Guide

This guide will help you migrate from the old configuration system to the new centralized configuration system.

## Overview

The new configuration system provides:
- Centralized configuration management
- Type-safe settings with validation
- Environment variable support
- Helper functions for common settings
- Enhanced security features
- Better organization of settings

## Migration Steps

### 1. Update Dependencies

Add the following to your `requirements.txt`:
```
pydantic>=2.0.0
python-dotenv>=1.0.0
```

### 2. Environment Variables

Create a `.env` file in your project root with the following structure:

```env
# Project Settings
PROJECT_NAME=Personal Assistant
VERSION=1.0.0
DEBUG=true
ENVIRONMENT=development

# API Settings
API_V1_STR=/api/v1
HOST=0.0.0.0
PORT=8000

# Database Settings
DATABASE_URL=sqlite:///./app.db

# Redis Settings
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your-redis-password

# Security Settings
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key-here
JWT_REFRESH_SECRET_KEY=your-refresh-secret-key-here

# Other settings...
```

### 3. Code Updates

#### Old Configuration:
```python
from backend.core.config.redis_config import RedisConfig
from backend.core.config.database_config import DatabaseConfig

redis_config = RedisConfig()
db_config = DatabaseConfig()
```

#### New Configuration:
```python
from backend.config import settings, get_redis_url, get_database_url

# Use settings directly
redis_url = get_redis_url()
db_url = get_database_url()
```

### 4. Service Updates

Update your services to use the new configuration:

#### Cache Service:
```python
from backend.config import settings, get_cache_settings

class CacheService:
    def __init__(self):
        self.cache_settings = get_cache_settings()
        self.ttl = self.cache_settings["ttl"]
        # ...
```

#### Search History Service:
```python
from backend.config import settings

class SearchHistoryService:
    def __init__(self):
        self.history_ttl = settings.SEARCH_HISTORY_TTL
        self.max_history = settings.MAX_HISTORY_PER_USER
        # ...
```

### 5. API Updates

Update your API endpoints to use the new configuration:

```python
from backend.config import settings, get_jwt_settings

@router.post("/login")
async def login():
    jwt_settings = get_jwt_settings()
    # ...
```

## New Features

### 1. Type-Safe Settings
All settings are now type-safe with validation:
```python
from backend.config import settings

# This will raise a validation error if the value is invalid
settings.PORT = "invalid"  # TypeError: invalid literal for int()
```

### 2. Environment Variable Support
Settings can be overridden using environment variables:
```bash
export PORT=8080
export DEBUG=false
```

### 3. Helper Functions
Use helper functions for common settings:
```python
from backend.config import get_redis_url, get_database_url, get_jwt_settings

redis_url = get_redis_url()
db_url = get_database_url()
jwt_settings = get_jwt_settings()
```

### 4. Security Features
- Production environment validation
- Password strength requirements
- JWT configuration
- CORS settings

## Best Practices

1. **Environment Variables**
   - Use environment variables for sensitive data
   - Never commit `.env` files to version control
   - Use different `.env` files for different environments

2. **Security**
   - Change default secret keys in production
   - Use strong passwords
   - Enable SSL/TLS where possible

3. **Validation**
   - Add custom validators for specific requirements
   - Use type hints for better IDE support
   - Validate settings on startup

4. **Organization**
   - Group related settings together
   - Use descriptive names
   - Document settings with comments

## Troubleshooting

### Common Issues

1. **Validation Errors**
   ```python
   # Error: Invalid database URL
   settings.DATABASE_URL = "invalid-url"
   ```

   Solution: Use a valid database URL format:
   ```python
   settings.DATABASE_URL = "sqlite:///./app.db"
   ```

2. **Environment Variables Not Loading**
   ```python
   # Error: Environment variable not found
   settings.REDIS_PASSWORD  # None
   ```

   Solution: Check your `.env` file and environment variables:
   ```bash
   export REDIS_PASSWORD=your-password
   ```

3. **Type Errors**
   ```python
   # Error: Invalid type
   settings.PORT = "8080"  # Should be int
   ```

   Solution: Use correct types:
   ```python
   settings.PORT = 8080
   ```

## Support

For additional help:
1. Check the [documentation](docs/README.md)
2. Open an issue on GitHub
3. Contact the development team

## Next Steps

1. Review your current configuration
2. Update your code to use the new system
3. Test your application
4. Deploy with the new configuration 