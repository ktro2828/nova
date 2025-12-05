# Nova Compression

## JPEG Encoder

The following table summarizes the supported JPEG encoders:

| Encoder                                | Backend                                                                             | Device |
| -------------------------------------- | ----------------------------------------------------------------------------------- | ------ |
| `nova::compression::CpuJpegEncoder`    | [TurboJPEG](https://github.com/libjpeg-turbo/libjpeg-turbo)                         | CPU    |
| `nova::compression::NvJpegEncoder`     | [nvJPEG](https://developer.nvidia.com/nvjpeg)                                       | GPU    |
| `nova::compression::JetsonJpegEncoder` | [JetsonJPEG](https://docs.nvidia.com/jetson/l4t-multimedia/classNvJPEGEncoder.html) | Jetson |
