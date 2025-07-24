# Use a lightweight Python base image
FROM python:3.10-slim

# Set working dir
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Launch the app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
