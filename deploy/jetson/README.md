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
  --output-json artifacts/predictions/jetson_benchmark.json
```

The executable reports preprocessing, TensorRT execution, post-processing, total mean, p50, p95 latency, FPS, warm-up count, and sample count. Run `tegrastats` concurrently for unified memory, power, and temperature, and keep Jetson power mode, clocks, JetPack, CUDA, and TensorRT versions fixed across variants.
