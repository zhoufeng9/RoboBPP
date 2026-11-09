# README media

These assets are taken from the RoboBPP project website and stored locally so the repository README does not depend on externally hosted media.

| File | Source | Processing |
| :--- | :--- | :--- |
| `robobpp-execution.gif` | [Execution Pack / Repetitive dataset](https://raw.githubusercontent.com/Robot-Bin-Packing-Benchmark/Robot-Bin-Packing-Benchmark.github.io/main/display/setting3/time_series.mp4) | Source interval 00:08–00:56, 3× playback speed, 800 × 600, 12 fps, 128-color palette, looping GIF |
| `overview.png` | [Project overview](https://raw.githubusercontent.com/Robot-Bin-Packing-Benchmark/Robot-Bin-Packing-Benchmark.github.io/main/display/overview.png) | Original image, unchanged |

The GIF shows simulation footage, not a physical robot experiment. Its playback speed is also stated in the main README caption.

To reproduce the GIF after downloading the source video as `execution.mp4`, run from the repository root:

```bash
ffmpeg -ss 8 -t 48 -i execution.mp4 -filter_complex '[0:v]setpts=(PTS-STARTPTS)/3,fps=12,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3' -loop 0 assets/robobpp-execution.gif
```
