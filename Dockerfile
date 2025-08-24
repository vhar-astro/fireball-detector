# Use Ubuntu 24.04 base image
FROM ubuntu:24.04

# Prevent interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy dependency file first for efficient caching
COPY requirements.txt .

# Copy project files into container
COPY . .

RUN python3 -m venv /opt/venv \
    source ~/venv/bin/activate

ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
# RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install -r torch_requirements_cpu.txt
RUN pip install pyinstaller



# Expose application port (update if your app uses a different port)
# EXPOSE 8000

# Default command (can be overridden in docker-compose)
# CMD ["python3", "main.py"]

ENTRYPOINT ["tail", "-f", "/dev/null"]
