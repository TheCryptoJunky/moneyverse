# Use the official Python 3.9 slim image for a lightweight setup
FROM python:3.9-slim

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
# These are necessary for certain Python libraries, such as cryptography and MySQL support
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libmysqlclient-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project to the container
COPY . .

# Expose any required ports (e.g., for API access, web access)
EXPOSE 5000  
# Example port for web-based APIs

# Set environment variables for MySQL, logging, trade API keys, and AI model paths
ENV MYSQL_HOST=${MYSQL_HOST}
ENV MYSQL_USER=${MYSQL_USER}
ENV MYSQL_PASSWORD=${MYSQL_PASSWORD}
ENV MYSQL_DATABASE=${MYSQL_DATABASE}
ENV LOGGING_METHOD=${LOGGING_METHOD}
ENV LOG_DB_TABLE=${LOG_DB_TABLE}
ENV LOG_LEVEL=${LOG_LEVEL}
ENV TRADE_API_KEY=${TRADE_API_KEY}
ENV TRADE_API_SECRET=${TRADE_API_SECRET}
ENV AI_MODEL_PATH=${AI_MODEL_PATH}

# Run the main bot application
CMD ["python", "main.py"]
