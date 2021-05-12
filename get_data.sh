#!/bin/bash
# Get Google Drive files here
# To view drive file, go to the link:
# https://drive.google.com/file/d/<file_id>

if [[ -d storage/external ]]; then
    echo "storage exists, starting download"
else
    mkdir --parents storage/external
fi

cd storage/external

if [[ ! -d "BGNN4VD" ]]; then
    git clone https://github.com/SicongCao/BGNN4VD.git
    cd BGNN4VD
    unzip dataset.zip
    mv dataset/* .
    rm -rf dataset .git dataset.zip README.md
    cd ..
else
    echo "Already downloaded BGNN4VD dataset"
fi

if [[ ! -d "reveal_chrome_debian" ]]; then
    gdown https://drive.google.com/uc\?id\=12CqPX5XdT0vKx4PWL2EKdGQbGrleKgJ-
    unzip reveal_chrome_debian.zip
    rm -rf reveal_chrome_debian.zip
else
    echo "Already downloaded ReVeal data"
fi

if [[ ! -d "devign_ffmpeg_qemu" ]]; then
    gdown https://drive.google.com/uc\?id\=1le__vMWFgsbPD_dWpwae5Ydu7EWwgYeu
    unzip devign_ffmpeg_qemu.zip
    rm -rf devign_ffmpeg_qemu.zip
else
    echo "Already downloaded Devign data"
fi

if [[ ! -d "w2v_models" ]]; then
    gdown https://drive.google.com/uc\?id\=1_vYrun3m1EUErTWwy3QiOmsiC3vyqgZa
    unzip w2v_models.zip
    rm -rf w2v_models.zip
else
    echo "Already downloaded w2v_models"
fi

cd ..