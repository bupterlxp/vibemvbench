# Vibe MV-Bench sandbox. Same image for all conditions; the condition only changes what is mounted/enabled.
FROM node:22-slim
ENV DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-cairo \
      ffmpeg chromium fonts-dejavu fonts-noto-core fonts-noto-cjk fonts-liberation \
      git curl ca-certificates imagemagick sox && rm -rf /var/lib/apt/lists/*
RUN npm install -g @openai/codex@0.162.1 @anthropic-ai/claude-code && npm cache clean --force
# numpy/scipy/Pillow come from pip in ONE resolver step (apt's python3-numpy/scipy were built for numpy 1.x, and opencv
# pulled numpy 2.x on top, which broke `import scipy` in 25/48 pilot runs). The smoke test fails the build if any
# advertised library cannot be imported.
RUN pip3 install --no-cache-dir "numpy>=2,<2.5" "scipy>=1.14" "pillow>=10" moviepy==2.1.2 imageio-ffmpeg opencv-python-headless && \
    python3 -c "import numpy, scipy, scipy.signal, scipy.fft, PIL, cairo, cv2, moviepy; print('sandbox imports ok', numpy.__version__, scipy.__version__)"
ENV CHROME_PATH=/usr/bin/chromium PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium
RUN useradd -m -u 1001 agent && mkdir -p /work && chown agent /work
WORKDIR /work
