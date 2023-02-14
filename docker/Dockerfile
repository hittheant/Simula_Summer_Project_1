FROM condaforge/mambaforge 

ENV DEBIAN_FRONTEND=noninteractive

# Install ssh (missing dependency to run conda envs)
RUN apt-get update && \
    apt-get install -y ssh build-essential

# Upgrade mamba
RUN mamba upgrade -y mamba

# Copy environment and requirements files into docker env
COPY environment.yml .

# Update environment file with new environment name
RUN mamba env update --file environment.yml --name dockerenv
SHELL ["mamba", "run", "-n", "dockerenv", "/bin/bash", "-c"]

# Install package
COPY . ffian/ 
RUN python3 -m pip install ./ffian

RUN echo "source activate dockerenv" > ~/.bashrc
