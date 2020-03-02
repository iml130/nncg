# NNCG: Neural Network Code Generator

## Table of Contents

- [Project Goals](#project-goals)
- [Current  Status](#current-status)
- [License](#license)
- [Contact](#contact)


## Project Goals

For the deployment of machine learning models on ressource constrained systems like mobile or IoT devices, usually a framework like TensorFlow Lite is utilized. Within this research project we tried to avoid any dependencies and thus decided to generate pure ANSI C code that can be added to existing projects for embedded devices. This simplifies the compilation, linking and deployment process. In addition, this allows to generate fast code as we can exploit the knowlegde about the model structure (size of matrices etc.) and about the target hardware (SIMD instructions etc.). Further details can be found in our paper:

* *"A C Code Generator for Fast Inference and Simple Deployment of Convolutional Neural Networks on Resource Constrained Systems"* available on [arXiv](https://arxiv.org/abs/2001.05572).


### C Code Example

This example demonstrates shows a code snipped of a machine learning modell. Since we know the structure of the model, we generate code for a specific target platform. For example SSE3 instructions are used to parallelize the calculation over the number of filters for a convolution layer. The generated code contains all weights as constants so that no array must be accessed. Loops can be unrolled and function calls are minimized to optimize cache accesses and also the CPU pipeline.

A full example can be found in `examples/daimler/daimler.c`.

```C
for (int i = 0; i < 1; i++)
    for (int j = 0; j < 1; j++)
        for (int k = 0; k < 2; k++){
            qx = cx7[i][j][k];              
            lo = _mm_srai_epi32(_mm_unpacklo_epi16(qx, qx), 16);
            hi = _mm_srai_epi32(_mm_unpackhi_epi16(qx, qx), 16);
            sum1 = _mm_hadd_epi32(hi, lo);
            sum2 = _mm_hadd_epi32(sum1, sum1);
            _mm_store_si128((__m128i*)res, sum2);
            x7[i][j][k] += (res[0] + res[1]) * 0.004617639413968784f * 0.008444292470812798f;
        }
static float x8[1][1][2] = {0};
static float max8 = 0;
max8 = x7[0][0][0] > x7[0][0][1] ? x7[0][0][0] : x7[0][0][1];
x8[0][0][0] = (float)exp(x7[0][0][0] - max8);
x8[0][0][1] = (float)exp(x7[0][0][1] - max8);
static float sum8;
sum8 = x8[0][0][0] + x8[0][0][1];
x8[0][0][0] /= sum8;
x8[0][0][1] /= sum8;
scores[0] = x8[0][0][0];
scores[1] = x8[0][0][1];
```


## Current Status

This is a research project, not ready for everyday use. The code is currently revised and it is intended to make NNCG available in **this** repository by end of March. Feel free to reach out by email if you require additional information prior to the public release.


## License

NNCG will be licensed under the terms of the Apache license. See LICENSE for more information.


## Contact

NNCG was initialy developed by Oliver Urbann and it's development is currently continued by the Fraunhofer Institute for Material Flow and Logistics.
