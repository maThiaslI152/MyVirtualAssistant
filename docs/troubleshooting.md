# Troubleshooting Guide

## Table of Contents
1. [Common Issues](#common-issues)
2. [Performance Issues](#performance-issues)
3. [Deployment Issues](#deployment-issues)
4. [Security Issues](#security-issues)
5. [Database Issues](#database-issues)
6. [Cache Issues](#cache-issues)
7. [API Issues](#api-issues)
8. [Frontend Issues](#frontend-issues)

## Common Issues

### Application Won't Start

#### Symptoms
- Application fails to start
- Error messages in logs
- Services not responding

#### Solutions
1. **Check Dependencies**
   ```bash
   # Backend
   pip install -r requirements.txt
   pip freeze > requirements.txt  # Update requirements

   # Frontend
   npm install
   npm audit fix  # Fix vulnerabilities
   ```

2. **Check Environment Variables**
   ```bash
   # Verify .env file exists
   ls -la .env

   # Check environment variables
   cat .env | grep -v "^#" | grep -v "^$"
   ```

3. **Check Port Availability**
   ```bash
   # Check if port is in use
   sudo lsof -i :8000
   sudo lsof -i :3000

   # Kill process if needed
   sudo kill -9 <PID>
   ```

### Service Connection Issues

#### Symptoms
- Services can't connect to each other
- Timeout errors
- Connection refused errors

#### Solutions
1. **Check Service Status**
   ```bash
   # Check Redis
   redis-cli ping
   systemctl status redis

   # Check Database
   psql -U postgres -c "\l"
   systemctl status postgresql
   ```

2. **Check Network**
   ```bash
   # Check localhost
   curl http://localhost:8000/health
   curl http://localhost:3000

   # Check network connectivity
   ping localhost
   telnet localhost 8000
   ```

3. **Check Firewall**
   ```bash
   # Check UFW status
   sudo ufw status

   # Check AWS Security Groups
   aws ec2 describe-security-groups
   ```

## Performance Issues

### Slow Response Times

#### Symptoms
- High latency
- Timeout errors
- Slow page loads

#### Solutions
1. **Check Resource Usage**
   ```bash
   # Check CPU usage
   top
   htop

   # Check memory usage
   free -m
   vmstat 1

   # Check disk usage
   df -h
   iostat 1
   ```

2. **Optimize Database**
   ```sql
   -- Check slow queries
   SELECT * FROM pg_stat_activity 
   WHERE state = 'active' 
   ORDER BY query_start DESC;

   -- Analyze tables
   ANALYZE table_name;
   ```

3. **Check Cache Performance**
   ```bash
   # Check Redis memory
   redis-cli info memory

   # Check cache hit rate
   redis-cli info stats | grep hit
   ```

### High Memory Usage

#### Symptoms
- Out of memory errors
- Slow performance
- Service crashes

#### Solutions
1. **Check Memory Leaks**
   ```bash
   # Check process memory
   ps aux | grep python
   ps aux | grep node

   # Check system memory
   free -m
   vmstat 1
   ```

2. **Optimize Application**
   ```python
   # Enable garbage collection
   import gc
   gc.collect()

   # Profile memory usage
   from memory_profiler import profile
   @profile
   def your_function():
       pass
   ```

3. **Adjust Resource Limits**
   ```bash
   # Adjust system limits
   sudo sysctl -w vm.max_map_count=262144
   sudo sysctl -w fs.file-max=65535
   ```

## Deployment Issues

### Docker Issues

#### Symptoms
- Container won't start
- Container crashes
- Network issues

#### Solutions
1. **Check Container Status**
   ```bash
   # List containers
   docker ps -a

   # Check container logs
   docker logs <container_id>

   # Check container resources
   docker stats
   ```

2. **Fix Common Issues**
   ```bash
   # Clean up containers
   docker system prune

   # Rebuild images
   docker-compose build --no-cache

   # Check network
   docker network ls
   docker network inspect <network_name>
   ```

3. **Debug Container**
   ```bash
   # Enter container
   docker exec -it <container_id> /bin/bash

   # Check processes
   ps aux

   # Check logs
   tail -f /var/log/application.log
   ```

### Cloud Deployment Issues

#### Symptoms
- Instance won't start
- Services unavailable
- Configuration errors

#### Solutions
1. **Check Instance Status**
   ```bash
   # AWS
   aws ec2 describe-instances --instance-ids <instance_id>

   # Google Cloud
   gcloud compute instances describe <instance_name>
   ```

2. **Check Cloud Logs**
   ```bash
   # AWS CloudWatch
   aws logs get-log-events --log-group-name /var/log/syslog

   # Google Cloud Logging
   gcloud logging read "resource.type=gce_instance"
   ```

3. **Verify Configuration**
   ```bash
   # Check security groups
   aws ec2 describe-security-groups

   # Check IAM roles
   aws iam get-role --role-name <role_name>
   ```

## Security Issues

### Authentication Issues

#### Symptoms
- Login failures
- Token errors
- Permission denied

#### Solutions
1. **Check JWT Configuration**
   ```python
   # Verify JWT settings
   from backend.config import get_jwt_settings
   jwt_settings = get_jwt_settings()
   print(jwt_settings)
   ```

2. **Check User Permissions**
   ```sql
   -- Check user roles
   SELECT * FROM users WHERE id = <user_id>;
   SELECT * FROM user_roles WHERE user_id = <user_id>;
   ```

3. **Verify Token**
   ```python
   # Decode and verify token
   from jose import jwt
   token_data = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
   ```

### API Security

#### Symptoms
- Unauthorized access
- Rate limit errors
- CORS issues

#### Solutions
1. **Check Rate Limiting**
   ```python
   # Verify rate limit settings
   from backend.config import settings
   print(settings.RATE_LIMIT_REQUESTS)
   print(settings.RATE_LIMIT_WINDOW)
   ```

2. **Check CORS Configuration**
   ```python
   # Verify CORS settings
   print(settings.CORS_ORIGINS)
   print(settings.CORS_METHODS)
   print(settings.CORS_HEADERS)
   ```

3. **Check API Keys**
   ```bash
   # Verify API key
   curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/health
   ```

## Database Issues

### Connection Issues

#### Symptoms
- Database connection errors
- Timeout errors
- Connection pool exhausted

#### Solutions
1. **Check Database Connection**
   ```python
   # Test connection
   from sqlalchemy import create_engine
   engine = create_engine(settings.DATABASE_URL)
   connection = engine.connect()
   ```

2. **Check Connection Pool**
   ```python
   # Monitor pool
   from sqlalchemy import event
   @event.listens_for(Engine, "checkout")
   def receive_checkout(dbapi_connection, connection_record, connection_proxy):
       print("Connection checked out")
   ```

3. **Optimize Pool Settings**
   ```python
   # Adjust pool settings
   engine = create_engine(
       settings.DATABASE_URL,
       pool_size=settings.DATABASE_POOL_SIZE,
       max_overflow=settings.DATABASE_MAX_OVERFLOW,
       pool_timeout=settings.DATABASE_POOL_TIMEOUT
   )
   ```

### Performance Issues

#### Symptoms
- Slow queries
- High CPU usage
- Memory issues

#### Solutions
1. **Analyze Queries**
   ```sql
   -- Check slow queries
   SELECT * FROM pg_stat_activity 
   WHERE state = 'active' 
   ORDER BY query_start DESC;

   -- Explain query
   EXPLAIN ANALYZE SELECT * FROM your_table;
   ```

2. **Optimize Indexes**
   ```sql
   -- Check indexes
   SELECT * FROM pg_indexes 
   WHERE tablename = 'your_table';

   -- Create index
   CREATE INDEX idx_column ON your_table(column);
   ```

3. **Vacuum Database**
   ```sql
   -- Vacuum analyze
   VACUUM ANALYZE your_table;

   -- Check table size
   SELECT pg_size_pretty(pg_total_relation_size('your_table'));
   ```

## Cache Issues

### Redis Issues

#### Symptoms
- Cache misses
- Slow responses
- Memory issues

#### Solutions
1. **Check Redis Status**
   ```bash
   # Check Redis info
   redis-cli info

   # Check memory usage
   redis-cli info memory

   # Check clients
   redis-cli client list
   ```

2. **Monitor Cache**
   ```python
   # Check cache hit rate
   from backend.services.cache_service import CacheService
   cache = CacheService()
   stats = cache.get_stats()
   print(stats)
   ```

3. **Optimize Cache**
   ```python
   # Adjust cache settings
   from backend.config import settings
   print(settings.CACHE_TTL)
   print(settings.CACHE_MAX_MEMORY)
   ```

## API Issues

### Endpoint Issues

#### Symptoms
- 404 errors
- 500 errors
- Timeout errors

#### Solutions
1. **Check API Documentation**
   ```bash
   # Check Swagger UI
   curl http://localhost:8000/docs

   # Check OpenAPI spec
   curl http://localhost:8000/openapi.json
   ```

2. **Test Endpoints**
   ```bash
   # Test health endpoint
   curl http://localhost:8000/health

   # Test API endpoint
   curl -X POST http://localhost:8000/api/v1/rag/search \
     -H "Content-Type: application/json" \
     -d '{"query": "test"}'
   ```

3. **Check Logs**
   ```bash
   # Check application logs
   tail -f /var/log/application.log

   # Check nginx logs
   tail -f /var/log/nginx/access.log
   tail -f /var/log/nginx/error.log
   ```

## Frontend Issues

### React Issues

#### Symptoms
- White screen
- Console errors
- Performance issues

#### Solutions
1. **Check Console**
   ```javascript
   // Enable debug logging
   console.debug('Debug message');
   console.error('Error message');
   ```

2. **Check Network**
   ```javascript
   // Monitor API calls
   fetch('/api/v1/health')
     .then(response => response.json())
     .then(data => console.log(data))
     .catch(error => console.error(error));
   ```

3. **Check State**
   ```javascript
   // Debug Redux state
   console.log(store.getState());

   // Debug React state
   console.log(this.state);
   ```

### Build Issues

#### Symptoms
- Build failures
- Missing dependencies
- Version conflicts

#### Solutions
1. **Check Dependencies**
   ```bash
   # Check npm dependencies
   npm ls

   # Check for vulnerabilities
   npm audit

   # Fix dependencies
   npm audit fix
   ```

2. **Clean Build**
   ```bash
   # Clean build
   rm -rf node_modules
   rm -rf build
   npm install
   npm run build
   ```

3. **Check Configuration**
   ```javascript
   // Check webpack config
   console.log(webpackConfig);

   // Check environment variables
   console.log(process.env);
   ``` 