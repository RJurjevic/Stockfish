#!/bin/bash
# makefish.sh

# install packages if not already installed
unzip -v &> /dev/null || pacman -S --noconfirm unzip
make -v &> /dev/null || pacman -S --noconfirm make
g++ -v &> /dev/null || pacman -S --noconfirm mingw-w64-x86_64-gcc

cd src

# build Stockfish executable
##make build ARCH=x86-64-avx2 blas=yes flipped=yes 
##make build ARCH=x86-64-avx2 blas=yes halphkp512=yes 
make build ARCH=x86-64-sse41-popcnt blas=yes halphkp512=yes 
##make build ARCH=x86-64-sse41-popcnt blas=yes flipped=yes 
##make build ARCH=x86-64-avx2 blas=yes flipped=yes -j
strip stockfish.exe
##mv stockfish.exe ../stockfish-sse41-flipped-2021-06-02.exe
##mv stockfish.exe ../stockfish-avx2-flipped-2021-06-04.exe
##mv stockfish.exe ../stockfish-avx2-halphkp512-2021-06-22.exe
mv stockfish.exe ../stockfish-sse41-halphkp512-2021-06-23.exe
make clean
cd
