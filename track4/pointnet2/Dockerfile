FROM tensorflow/tensorflow:1.12.0-gpu-py3
MAINTAINER JHUAPL <pubgeo@jhuapl.edu>

# Install wget
RUN apt install -y --fix-missing --no-install-recommends wget

# Install conda
RUN mkdir ~/Downloads \
    && wget https://repo.anaconda.com/miniconda/Miniconda2-4.5.11-Linux-x86_64.sh -nv -O ~/Downloads/Miniconda2-4.5.11-Linux-x86_64.sh
RUN bash ~/Downloads/Miniconda2-4.5.11-Linux-x86_64.sh -b -p /opt/conda

# Install pdal as a conda environment
RUN /opt/conda/bin/conda create --yes --name cpdal-run --channel conda-forge pdal=1.7

# Compile and install pointnet2 tensorflow utilities
COPY tf_ops /pointnet2/tf_ops
RUN cd /pointnet2/tf_ops/3d_interpolation && \
    chmod +x ./tf_interpolate_compile_py3.sh && chmod +x ./tf_interpolate_compile.sh && \
    ./tf_interpolate_compile_py3.sh && \
    cd /pointnet2/tf_ops/grouping && \
    chmod +x ./tf_grouping_compile_py3.sh && chmod +x ./tf_grouping_compile.sh && \
    ./tf_grouping_compile_py3.sh && \
    cd /pointnet2/tf_ops/sampling && \
    chmod +x ./tf_sampling_compile_py3.sh && chmod +x ./tf_sampling_compile.sh && \
    ./tf_sampling_compile_py3.sh

# Set working directory
WORKDIR /pointnet2

# Install laspy, pathlib
RUN apt install -y --fix-missing --no-install-recommends python3-pip
RUN pip3 install laspy numpy
