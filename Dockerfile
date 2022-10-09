FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# Install base utilities
RUN apt-get update && \
    apt-get install python3 -y && \
    apt-get install python3-pip -y &&  \
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 && \
    pip3 install boto3 diffusers==0.3.0 transformers scipy ftfy

ADD main.py /

CMD ["python3", "./main.py"]