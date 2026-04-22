#!/usr/bin/env python3

"""
CSA Setup and Integration Helper
Helps users set up CSA with different inference engines
"""

import subprocess
import sys
import os
import platform

class CSASetup:
    def __init__(self):
        self.system = platform.system().lower()

    def run_command(self, cmd, description=""):
        """Run a command and handle errors"""
        try:
            print(f"🔧 {description}")
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed: {e.stderr}")
            return False

    def check_dependency(self, name, command):
        """Check if a dependency is available"""
        try:
            subprocess.run(command, shell=True, check=True, capture_output=True)
            return True
        except:
            return False

    def setup_ollama(self):
        """Set up Ollama integration"""
        print("\n🦙 Setting up Ollama Integration")
        print("-" * 40)

        # Check if Ollama is installed
        if not self.check_dependency("Ollama", "ollama --version"):
            print("📥 Installing Ollama...")

            if self.system == "linux":
                self.run_command(
                    "curl -fsSL https://ollama.ai/install.sh | sh",
                    "Installing Ollama for Linux"
                )
            elif self.system == "darwin":  # macOS
                self.run_command(
                    "/bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"",
                    "Installing Homebrew (if needed)"
                )
                self.run_command("brew install ollama", "Installing Ollama via Homebrew")
            else:
                print("❌ Please install Ollama manually from https://ollama.ai")
                return False

        # Start Ollama service
        print("🚀 Starting Ollama service...")
        if self.system == "windows":
            # On Windows, user needs to start manually
            print("📝 On Windows, please start Ollama manually:")
            print("   1. Open Command Prompt as Administrator")
            print("   2. Run: ollama serve")
        else:
            # Try to start service in background
            self.run_command("ollama serve", "Starting Ollama service")

        # Pull a model for testing
        print("📥 Pulling Llama2 model for testing...")
        self.run_command("ollama pull llama2", "Downloading Llama2 model")

        print("✅ Ollama setup complete!")
        print("🧪 Test with: python integration_examples.py")
        return True

    def setup_vllm(self):
        """Set up vLLM integration"""
        print("\n🚀 Setting up vLLM Integration")
        print("-" * 40)

        # Install vLLM (it was already in pyproject.toml, but let's verify)
        print("📦 Installing vLLM...")
        if not self.run_command("pip install vllm", "Installing vLLM"):
            print("❌ vLLM installation failed. It requires CUDA and can be complex to install.")
            print("💡 Alternative: Use the CPU-only version or skip vLLM integration for now.")
            return False

        print("✅ vLLM installed successfully!")
        print("🧪 To test vLLM:")
        print("   python -m vllm.entrypoints.openai.api_server --model gpt2 --host 0.0.0.0 --port 8000")
        print("   Then run: python integration_examples.py")
        return True

    def setup_csa(self):
        """Set up CSA package"""
        print("\n⚡ Setting up CSA Package")
        print("-" * 40)

        # Install CSA
        if not self.run_command("pip install -e .", "Installing CSA package"):
            print("❌ CSA installation failed")
            return False

        # Test installation
        if not self.run_command("python -c \"from csa import CSAEngine; print('CSA imported successfully')\"", "Testing CSA import"):
            print("❌ CSA import failed")
            return False

        print("✅ CSA setup complete!")
        return True

    def generate_startup_script(self):
        """Generate a startup script for easy launching"""
        print("\n📝 Generating startup script...")

        script_content = '''#!/bin/bash
# CSA Integration Startup Script

echo "🚀 Starting CSA Integration Services"

# Start CSA integration server
echo "Starting CSA server on port 5000..."
python integration_server.py &
CSA_PID=$!

# Optionally start Ollama (uncomment if installed)
# echo "Starting Ollama..."
# ollama serve &
# OLLAMA_PID=$!

# Optionally start vLLM (uncomment if installed)
# echo "Starting vLLM server..."
# python -m vllm.entrypoints.openai.api_server --model gpt2 --host 0.0.0.0 --port 8000 &
# VLLM_PID=$!

echo "Services started!"
echo "CSA Server: http://localhost:5000"
echo "Ollama: http://localhost:11434"
echo "vLLM: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "echo 'Stopping services...'; kill $CSA_PID 2>/dev/null; exit" INT
wait
'''

        with open('start_services.sh', 'w') as f:
            f.write(script_content)

        # Make executable on Unix-like systems
        if self.system != "windows":
            os.chmod('start_services.sh', 0o755)

        print("✅ Created start_services.sh")
        print("🧪 Run with: ./start_services.sh")

    def create_docker_setup(self):
        """Create Docker setup for easy deployment"""
        print("\n🐳 Creating Docker setup...")

        dockerfile = '''FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install Ollama (optional)
RUN curl -fsSL https://ollama.ai/install.sh | sh || true

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy CSA code
COPY . .

# Install CSA
RUN pip install -e .

# Expose ports
EXPOSE 5000 11434 8000

# Default command
CMD ["python", "integration_server.py"]
'''

        with open('Dockerfile', 'w') as f:
            f.write(dockerfile)

        docker_compose = '''version: '3.8'

services:
  csa-server:
    build: .
    ports:
      - "5000:5000"
    environment:
      - PYTHONPATH=/app
    volumes:
      - .:/app
    command: python integration_server.py

  # Optional: Ollama service
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama:/root/.ollama
    command: serve

  # Optional: vLLM service
  vllm:
    build:
      context: .
      dockerfile: Dockerfile.vllm
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  ollama:
'''

        with open('docker-compose.yml', 'w') as f:
            f.write(docker_compose)

        print("✅ Created Dockerfile and docker-compose.yml")
        print("🐳 Build with: docker-compose build")
        print("🚀 Run with: docker-compose up")

    def interactive_setup(self):
        """Interactive setup wizard"""
        print("🎯 CSA Integration Setup Wizard")
        print("=" * 50)

        # CSA setup (required)
        if input("Set up CSA package? (y/n): ").lower().startswith('y'):
            self.setup_csa()

        # Ollama setup
        if input("Set up Ollama integration? (y/n): ").lower().startswith('y'):
            self.setup_ollama()

        # vLLM setup
        if input("Set up vLLM integration? (y/n): ").lower().startswith('y'):
            self.setup_vllm()

        # Additional tools
        if input("Generate startup scripts and Docker setup? (y/n): ").lower().startswith('y'):
            self.generate_startup_script()
            self.create_docker_setup()

        print("\n🎉 Setup complete!")
        print("📚 Check integration_guide.md for usage instructions")
        print("🧪 Run integration_examples.py to test integrations")

def main():
    setup = CSASetup()

    if len(sys.argv) > 1:
        # Command line mode
        command = sys.argv[1].lower()

        if command == "csa":
            setup.setup_csa()
        elif command == "ollama":
            setup.setup_ollama()
        elif command == "vllm":
            setup.setup_vllm()
        elif command == "docker":
            setup.create_docker_setup()
        else:
            print("Usage: python setup.py [csa|ollama|vllm|docker]")
    else:
        # Interactive mode
        setup.interactive_setup()

if __name__ == "__main__":
    main()