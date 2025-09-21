FROM python:3.10-slim
# FROM ubuntu:latest

WORKDIR /home/poojith-xyz/projects/ait

# RUN apt-get update
# RUN apt-get install -y python3 python3-pip
# RUN apt-get install python3.10
# RUN apt-get install -y libgomp1

# RUN apt-get install -y pipx && pipx install virtualenv && pipx ensurepath && \
#     PATH=/root/.local/bin:$PATH virtualenv -p python3.10 /home/poojith-xyz/projects/ait

RUN rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "src/home.py"]
