FROM python:3.12-slim

LABEL maintainer="Corey A. Wade <corey@coreywade.com>"
LABEL description="InfraWatch — Production anomaly detection for infrastructure metrics"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install package
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY sample_data/ sample_data/

RUN pip install --no-cache-dir -e ".[ml]"

# Non-root user
RUN useradd --create-home infrawatch
USER infrawatch

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

ENTRYPOINT ["infrawatch"]
CMD ["dashboard", "--host", "0.0.0.0", "--port", "8080"]
