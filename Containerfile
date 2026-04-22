# Use Red Hat UBI9 Python 3.11 — optimized for OpenShift
FROM registry.access.redhat.com/ubi9/python-311:latest

# Metadata
LABEL name="ai-demo-scratchpad" \
      description="Redis + OpenShift AI demo: semantic caching, routing, RAG, and vector search" \
      maintainer="cmays"

WORKDIR /app

# Install dependencies first (layer cache — only rebuilt when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# OpenShift runs pods as a random UID in the root group (GID 0).
# Switch to root to chgrp, then drop back to the image's default non-root user.
USER 0
RUN chgrp -R 0 /app && chmod -R g=u /app
USER 1001

# Streamlit runtime configuration
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501"]
