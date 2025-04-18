FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

WORKDIR /project

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y --no-install-recommends git build-essential

# install python
RUN apt install -y --no-install-recommends python3.12 python3.12-dev python3-pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# fix pip, https://stackoverflow.com/questions/75608323/how-do-i-solve-error-externally-managed-environment-every-time-i-use-pip-3
RUN python -m pip config set global.break-system-packages true

# these are needed for opencv, i believe
RUN apt install -y --no-install-recommends libgl1 libglib2.0-0

# install required python packages
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
RUN pip install \
    opencv-python==4.11.0.86 \
    huggingface-hub==0.30.2 \
    Cython==3.0.12 \
    einops==0.8.1 \
    smplx==0.1.28 \
    scipy==1.15.2

# install the latest chumpy
RUN pip install git+https://github.com/mattloper/chumpy.git@51d5afd92a8ded3637553be8cef41f328a1c863a

# a bug fix is needed for chumpy, https://github.com/mattloper/chumpy/issues/57
RUN sed -i 's/inspect.getargspec/inspect.getfullargspec/g' /usr/local/lib/python3.12/dist-packages/chumpy/ch.py
