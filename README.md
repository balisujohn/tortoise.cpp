# GGML implementation of tortoise-tts, under construction
Implementation status:

Tokenization seems to work, but doesn't exactly match the tokenization tortoise-tts performs, needs work. 

Voice latent is hardcoded for now. 

Text embedding/ text position embedding reconstruction complete with numbers matching.

Mel embedding reconstruction complete with numbers matching (at least for initial embedding).

Autoregressive model(gpt-2) reconstruction in progress.

Diffusion model reconstruction pending. 


# Compiling
For now, cuda only. To compile:
````
mkdir build
cd build
cmake -DGGML_CUBLAS=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc ..
make
````
This is tested with Ubuntu 22.04 and cuda 12.0 and a 1070ti


# Running
You will need to place `ggml-model.bin` in the examples/tortoise.directory to run tortoise.cpp. You can generate the model yourself following the instructions in this tortoise-tts reverse engineering fork here https://github.com/balisujohn/tortoise-reverse-engineering, or download it here https://huggingface.co/balisujohn/tortoise-ggml.


From the build directory, run:
````
./bin/tortoise
````


# Contributing
If you want to contribute, please make an issue stating what you want to work on. I'll make a discord to manage contributors if there is a lot of interest. You can email me questions at \<mylastname\>u\<myfirstname\>@gmail.com. I am happy to help get people get started with contributing!

I am also making available a fork of tortoise-tts which has my reverse engineering annotations, and also the export script for the autoregressive model.

# License

This is released with an MIT License.

Derived from tortoise-tts and ggml.

## tortoise-tts:
Apache 2.0 License James Betker
https://github.com/neonbjb/tortoise-tts/blob/main/LICENSE

## GGML
MIT License

Copyright (c) 2022 Georgi Gerganov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
