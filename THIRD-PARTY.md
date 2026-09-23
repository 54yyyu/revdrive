# Third-party material

- **rev** (MIT, same author) reads the served model: `rev.remote.Remote.read_tokens`.
- **moderngl** (MIT) draws the cameras headless; **NumPy** (BSD-3-Clause) and
  **Pillow** (MIT-CMU) are the rest of the runtime.
- The web view loads **IBM Plex** Sans, Serif and Mono from Google Fonts (SIL Open
  Font License 1.1); nothing is bundled.
- **DrivingBench** (drivingbench.com; harness at
  github.com/aditya-ramabadran/drivingbench_harness_v1) is the reference this was
  aligned with: the camera mount, the cone density (read off their course map),
  the speed range, the rule that a touched cone ends the run, and course 100's
  layout (from their report's description). No DrivingBench code, prompt text,
  images or data are included. revdrive is not affiliated with DrivingBench.
- The simulated car's dimensions and turning circle are a Toyota Corolla's
  (public specifications); no manufacturer material is used.
- The served model in the published results is Qwen3.8-27B, quantized AWQ INT4
  (its license is on its model card); the model is not included.
