# Jetson Orin Nano benchmark

Build this target on the Jetson, where CUDA and TensorRT are installed:

```bash
cmake -S deploy/jetson -B build/jetson
cmake --build build/jetson --config Release -j2
```

Run batch-1 measurements with a warmed-up engine:

```bash
./build/jetson/jetson_benchmark \
  --engine artifacts/tensorrt/model.engine \
  --input-dir data/camel/images/test \
  --warmup 20 \
  --iterations 100 \
  --raw-output-dir artifacts/predictions/jetson_raw_outputs \
  --output-json artifacts/predictions/jetson_benchmark.json
```

The executable reads real `.jpg`, `.jpeg`, and `.png` images, resizes grayscale infrared images to the fixed engine input shape, normalizes them to `[0, 1]`, and replicates the channel when the engine expects three channels. It reports preprocessing, host-to-device transfer, TensorRT execution, device-to-host transfer, raw-output serialization, total mean, p50, p95 latency, FPS, warm-up count, image count, and measured sample count.

`--raw-output-dir` is optional. When supplied, it writes output tensors as binary files and writes `metadata.json` describing the source images and tensor shapes. These raw outputs are intended for Python-side decoding and mAP evaluation; this executable is the authoritative path for Jetson performance measurements, not detection-quality scoring.

Run `tegrastats` concurrently for unified memory, power, and temperature, for example:

```bash
tegrastats --interval 1000 | tee artifacts/predictions/tegrastats.log
```

Keep Jetson power mode, clocks, JetPack, CUDA, TensorRT, OpenCV, input resolution, warm-up, and iteration counts fixed across variants. For pure engine-performance measurements, omit `--raw-output-dir` so disk serialization is not included in end-to-end latency.
