# Deployment Guide

## Table of Contents
1. [Development Environment](#development-environment)
2. [Production Environment](#production-environment)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Scaling](#scaling)
6. [Monitoring](#monitoring)
7. [Backup and Recovery](#backup-and-recovery)

## Development Environment

### Local Setup

1. **Prerequisites**
   ```bash
   # Install Python 3.8+
   python --version

   # Install Node.js 16+
   node --version

   # Install Redis
   redis-server --version
   ```

2. **Backend Setup**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   cp .env.example .env
   ```

4. **Start Services**
   ```bash
   # Start Redis
   redis-server

   # Start Backend
   cd backend
   uvicorn main:app --reload

   # Start Frontend
   cd frontend
   npm run dev
   ```

### Development with Docker

1. **Build Development Images**
   ```bash
   docker-compose -f docker-compose.dev.yml build
   ```

2. **Start Development Environment**
   ```bash
   docker-compose -f docker-compose.dev.yml up
   ```

## Production Environment

### Traditional Deployment

1. **Server Requirements**
   - Linux (Ubuntu 20.04+ recommended)
   - 4+ CPU cores
   - 8GB+ RAM
   - 50GB+ storage
   - SSL certificate

2. **System Setup**
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y

   # Install dependencies
   sudo apt install -y python3.8 python3.8-venv nginx redis-server

   # Install Node.js
   curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
   sudo apt install -y nodejs
   ```

3. **Application Setup**
   ```bash
   # Clone repository
   git clone https://github.com/yourusername/personal-assistant.git
   cd personal-assistant

   # Setup backend
   cd backend
   python3.8 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   # Edit .env with production settings

   # Setup frontend
   cd ../frontend
   npm install
   npm run build
   cp .env.example .env
   # Edit .env with production settings
   ```

4. **Nginx Configuration**
   ```nginx
   # /etc/nginx/sites-available/personal-assistant
   server {
       listen 80;
       server_name your-domain.com;
       return 301 https://$server_name$request_uri;
   }

   server {
       listen 443 ssl;
       server_name your-domain.com;

       ssl_certificate /path/to/cert.pem;
       ssl_certificate_key /path/to/key.pem;

       location / {
           root /path/to/frontend/dist;
           try_files $uri $uri/ /index.html;
       }

       location /api {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }

       location /ws {
           proxy_pass http://localhost:8000;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
       }
   }
   ```

5. **Systemd Service**
   ```ini
   # /etc/systemd/system/personal-assistant.service
   [Unit]
   Description=Personal Assistant Backend
   After=network.target

   [Service]
   User=www-data
   Group=www-data
   WorkingDirectory=/path/to/backend
   Environment="PATH=/path/to/backend/venv/bin"
   ExecStart=/path/to/backend/venv/bin/gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

6. **Start Services**
   ```bash
   sudo systemctl enable personal-assistant
   sudo systemctl start personal-assistant
   sudo systemctl restart nginx
   ```

## Docker Deployment

### Production Docker Setup

1. **Docker Compose Configuration**
   ```yaml
   # docker-compose.yml
   version: '3.8'

   services:
     backend:
       build: 
         context: ./backend
         dockerfile: Dockerfile.prod
       environment:
         - REDIS_URL=redis://redis:6379
       depends_on:
         - redis
       restart: always

     frontend:
       build:
         context: ./frontend
         dockerfile: Dockerfile.prod
       ports:
         - "80:80"
       depends_on:
         - backend
       restart: always

     redis:
       image: redis:alpine
       volumes:
         - redis_data:/data
       restart: always

   volumes:
     redis_data:
   ```

2. **Backend Dockerfile**
   ```dockerfile
   # backend/Dockerfile.prod
   FROM python:3.8-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
   ```

3. **Frontend Dockerfile**
   ```dockerfile
   # frontend/Dockerfile.prod
   FROM node:16-alpine as builder

   WORKDIR /app

   COPY package*.json ./
   RUN npm install

   COPY . .
   RUN npm run build

   FROM nginx:alpine
   COPY --from=builder /app/dist /usr/share/nginx/html
   COPY nginx.conf /etc/nginx/conf.d/default.conf
   ```

4. **Deploy**
   ```bash
   docker-compose up -d
   ```

## Cloud Deployment

### AWS Deployment

1. **EC2 Setup**
   ```bash
   # Launch EC2 instance
   aws ec2 run-instances \
     --image-id ami-0c55b159cbfafe1f0 \
     --instance-type t2.medium \
     --key-name your-key-pair \
     --security-group-ids sg-xxxxxxxx
   ```

2. **RDS Setup**
   ```bash
   # Create RDS instance
   aws rds create-db-instance \
     --db-instance-identifier personal-assistant \
     --db-instance-class db.t3.micro \
     --engine postgres \
     --master-username admin \
     --master-user-password your-password
   ```

3. **ElastiCache Setup**
   ```bash
   # Create ElastiCache cluster
   aws elasticache create-cache-cluster \
     --cache-cluster-id personal-assistant \
     --cache-node-type cache.t3.micro \
     --engine redis \
     --num-cache-nodes 1
   ```

4. **S3 Setup**
   ```bash
   # Create S3 bucket
   aws s3 mb s3://personal-assistant-files
   ```

### Google Cloud Deployment

1. **Compute Engine Setup**
   ```bash
   # Create VM instance
   gcloud compute instances create personal-assistant \
     --machine-type e2-medium \
     --image-family ubuntu-2004-lts \
     --image-project ubuntu-os-cloud
   ```

2. **Cloud SQL Setup**
   ```bash
   # Create Cloud SQL instance
   gcloud sql instances create personal-assistant \
     --database-version POSTGRES_13 \
     --tier db-f1-micro
   ```

3. **Cloud Memorystore Setup**
   ```bash
   # Create Redis instance
   gcloud redis instances create personal-assistant \
     --size=1 \
     --region=us-central1
   ```

4. **Cloud Storage Setup**
   ```bash
   # Create storage bucket
   gsutil mb gs://personal-assistant-files
   ```

## Scaling

### Horizontal Scaling

1. **Load Balancer Setup**
   ```bash
   # AWS ELB
   aws elb create-load-balancer \
     --load-balancer-name personal-assistant \
     --listeners Protocol=HTTP,LoadBalancerPort=80,InstanceProtocol=HTTP,InstancePort=80 \
     --subnets subnet-xxxxxxxx
   ```

2. **Auto Scaling Group**
   ```bash
   # AWS Auto Scaling Group
   aws autoscaling create-auto-scaling-group \
     --auto-scaling-group-name personal-assistant \
     --launch-configuration-name personal-assistant-lc \
     --min-size 2 \
     --max-size 10 \
     --desired-capacity 2
   ```

### Vertical Scaling

1. **Database Scaling**
   ```bash
   # AWS RDS Scaling
   aws rds modify-db-instance \
     --db-instance-identifier personal-assistant \
     --db-instance-class db.t3.large
   ```

2. **Cache Scaling**
   ```bash
   # AWS ElastiCache Scaling
   aws elasticache modify-cache-cluster \
     --cache-cluster-id personal-assistant \
     --cache-node-type cache.t3.large
   ```

## Monitoring

### Prometheus Setup

1. **Install Prometheus**
   ```bash
   # Download Prometheus
   wget https://github.com/prometheus/prometheus/releases/download/v2.30.0/prometheus-2.30.0.linux-amd64.tar.gz
   tar xvfz prometheus-*.tar.gz
   cd prometheus-*
   ```

2. **Configure Prometheus**
   ```yaml
   # prometheus.yml
   global:
     scrape_interval: 15s

   scrape_configs:
     - job_name: 'personal-assistant'
       static_configs:
         - targets: ['localhost:8000']
   ```

3. **Start Prometheus**
   ```bash
   ./prometheus --config.file=prometheus.yml
   ```

### Grafana Setup

1. **Install Grafana**
   ```bash
   # Add Grafana repository
   sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
   sudo apt update
   sudo apt install grafana
   ```

2. **Configure Grafana**
   ```bash
   # Start Grafana
   sudo systemctl start grafana-server
   sudo systemctl enable grafana-server
   ```

3. **Add Prometheus Data Source**
   - Open Grafana UI (http://localhost:3000)
   - Add Prometheus data source
   - Configure dashboards

## Backup and Recovery

### Database Backup

1. **PostgreSQL Backup**
   ```bash
   # Create backup
   pg_dump -U postgres personal_assistant > backup.sql

   # Restore backup
   psql -U postgres personal_assistant < backup.sql
   ```

2. **Redis Backup**
   ```bash
   # Create backup
   redis-cli SAVE

   # Restore backup
   redis-cli FLUSHALL
   redis-cli RESTORE key 0 value
   ```

### File Backup

1. **S3 Backup**
   ```bash
   # Backup to S3
   aws s3 sync /path/to/files s3://personal-assistant-files/backup

   # Restore from S3
   aws s3 sync s3://personal-assistant-files/backup /path/to/files
   ```

2. **Local Backup**
   ```bash
   # Create backup
   tar -czf backup.tar.gz /path/to/files

   # Restore backup
   tar -xzf backup.tar.gz -C /path/to/restore
   ```

### Disaster Recovery

1. **Create Recovery Plan**
   ```bash
   # Backup all data
   ./scripts/backup.sh

   # Test recovery
   ./scripts/recover.sh
   ```

2. **Automated Recovery**
   ```bash
   # Setup automated backups
   crontab -e
   # Add: 0 0 * * * /path/to/scripts/backup.sh
   ```

## Security

### SSL/TLS Setup

1. **Let's Encrypt**
   ```bash
   # Install Certbot
   sudo apt install certbot python3-certbot-nginx

   # Get certificate
   sudo certbot --nginx -d your-domain.com
   ```

2. **Auto-renewal**
   ```bash
   # Test auto-renewal
   sudo certbot renew --dry-run
   ```

### Firewall Setup

1. **UFW Configuration**
   ```bash
   # Allow SSH
   sudo ufw allow ssh

   # Allow HTTP/HTTPS
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp

   # Enable firewall
   sudo ufw enable
   ```

2. **Security Groups (AWS)**
   ```bash
   # Create security group
   aws ec2 create-security-group \
     --group-name personal-assistant \
     --description "Personal Assistant Security Group"

   # Add rules
   aws ec2 authorize-security-group-ingress \
     --group-name personal-assistant \
     --protocol tcp \
     --port 80 \
     --cidr 0.0.0.0/0
   ``` 