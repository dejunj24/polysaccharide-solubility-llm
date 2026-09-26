FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

# 시스템 기본 설정 및 필수 패키지
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-distutils \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 기본 링크 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

# 파이썬 의존성 설치
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip "setuptools<70"
RUN pip install --no-cache-dir torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install --no-cache-dir -r requirements.txt

# 모델 아티팩트 및 서빙 소스 복사
COPY models /app/models
COPY src /app/src

EXPOSE 8000

# FastAPI 서버 구동
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
