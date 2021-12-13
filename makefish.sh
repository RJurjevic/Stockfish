#!/bin/bash
# makefish.sh

# install packages if not already installed
unzip -v &> /dev/null || pacman -S --noconfirm unzip
make -v &> /dev/null || pacman -S --noconfirm make
g++ -v &> /dev/null || pacman -S --noconfirm mingw-w64-x86_64-gcc

cd src

# build Stockfish executable
make build ARCH=x86-64-avx2 blas=yes flipped=yes
strip stockfish.exe
mv stockfish.exe ../stockfish-avx2-halfkp-2021-12-13.exe
make clean
cd
