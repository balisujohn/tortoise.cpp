# tortoise.cpp: GGML implementation of tortoise-tts ([Support Development Here!](https://ko-fi.com/johnbalis))

![a menacing sea turtle in the ocean; mascot for tortoise.cpp](https://github.com/balisujohn/tortoise.cpp/blob/master/assets/tortoiselogo.png?raw=true)

# Downloading
clone the repository with the following command
````
git clone --recursive https://github.com/balisujohn/tortoise.cpp.git
````
# Compiling
For now, CUDA and CPU only. To compile:

## Compile for CPU (works on Linux x86 and Mac ARM)
````
mkdir build
cd build
cmake .. 
make
````
This is tested with mac os arm

## Compile for CUDA
````
mkdir build
cd build
cmake .. -DGGML_CUDA=ON
make
````
This is tested with Ubuntu 22.04 and cuda 12.0 and a 1070ti

## Compile for Mac OS with metal (work in-progress)
````
mkdir build
cd build
cmake .. -DGGML_METAL=ON
make
````

# Running

**Only lowercase letters, spaces, and punctuation are supported in the prompt.**

You will need to place `ggml-model.bin`, `ggml-vocoder-model.bin` and `ggml-diffusion-model.bin` in the models directory to run tortoise.cpp. You can download them here https://huggingface.co/balisujohn/tortoise-ggml. I will release scripts for generating these files from tortoise-tts.

From the build directory, run:
````
./tortoise
````
here's an example that should work out of the box:
````
./tortoise --message "based... dr freeman?" --voice "../models/mouse.bin" --seed 0 --output "based?.wav"
````
all command line arguments are optional:

````
arguments:
  --message           Specifies the message to generate, lowercase letters, spaces, and punctuation only. (default: "this is a test message." )
  --voice             Specifies the path to the voice file to use to determine the speaker's voice.  (default: "../models/mol.bin" )
  --output            Specifies the path where the generated wav file will be saved.                 (default: "./output.wav")
  --seed              Specifies the seed for psuedorandom number generation, used in autoregressive sampling and diffusion sampling (default: system time seed)
````

# How to add voices

set up the original tortoise-tts, then run it with whatever voice you have, then after this line:
https://github.com/neonbjb/tortoise-tts/blob/e2d9fba0bb5c4376d0d142efea47a448f97c4d90/tortoise/api.py#L401

add this code:
````
numpy_array = auto_conditioning.to("cpu").numpy().astype(np.float32)  # Ensure float32 for binary format

# Define the file path
file_path = 'auto_conditioning.bin'

# Save NumPy array as binary file
numpy_array.tofile(file_path)

print("saved auto conditioning")
exit()
````
then you can rename `auto_conditioning.bin` to the speaker name and put the file in your models folder to use it like any other voice. This works with voices clone with `tortoise-tts`.

# Contributing
If you want to contribute, please make an issue stating what you want to work on. DM me on twitter if you want a link to join the dev Discord, or if you have questions. I am happy to help get people get started with contributing!

I am also making available a fork of tortoise-tts which has my reverse engineering annotations, and also the export script for the autoregressive model.

# License

This is released with an MIT License.

MIT License

Copyright (c) 2024 John Balis

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
