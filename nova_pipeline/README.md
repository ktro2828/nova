# Nova Pipeline

## Rectifier

The following table summarizes the supported rectifiers:

| Rectifier                            | Backend                                                                            | Device |
| ------------------------------------ | ---------------------------------------------------------------------------------- | ------ |
| `nova::pipeline::OpenCVRectifierCPU` | [OpenCV](https://opencv.org/)                                                      | CPU    |
| `nova::pipeline::OpenCVRectifierGPU` | [OpenCV CUDA](https://opencv.org/platforms/cuda/)                                  | GPU    |
| `nova::pipeline::NPPRectifier`       | [NVIDIA Performance Primitives (NPP)](https://docs.nvidia.com/cuda/npp/index.html) | GPU    |
