# Single-Video Playback Performance — Why It Was Slow

> **YOLO Vision Studio** · the `_run_video_loop()` path (one video selected).
> For the N-video path see [`multi_video_performance.md`](multi_video_performance.md).
>
> Measured on an NVIDIA RTX PRO 4500 Blackwell, ultralytics 8.4.116, torch 2.8.0+cu128,
> `yolo26n.pt`, on the clips in `videos/`. Server-side numbers come from Streamlit's own
> `AppTest` harness; anything extrapolated says so.

**Inference was never the bottleneck.** YOLO26-nano runs a 640×360 frame in 9.5 ms — 105 fps,
three times faster than the video plays. Everything that made playback feel slow happened
after the model returned.

---

## The clue: two videos played faster than one

That should be impossible if the GPU is the constraint, because two videos is strictly more
inference. So the constraint was elsewhere. Counting the messages the server pushes to the
browser (via `ForwardMsgQueue._before_enqueue_msg`) found it:

| Path | ForwardMsgs per displayed frame | msgs/s |
|---|---|---|
| Single video, original | **8.32** | 388 |
| Two videos, original | 2.05 | 138 |
| Single video, fixed | **0.88** | ~60 |

The single-video path redrew **seven sidebar placeholders on every frame** — `Frame`, `FPS`,
`Objects in Frame`, the local class list, `Total Unique Objects`, the global class list and
the churn line. The multi-video path drew one markdown line per video. One video therefore
made the browser do four times more work per frame than two videos did, and the video —
the thing you were actually watching — queued behind widget updates nobody can read at
90 fps.

---

## Where the time goes now

Profiled over 400 frames of `people_crossing_1`, stage by stage:

```
400 frames, wall 5.16s -> 77.6 fps, 12.89 ms/frame

  decode           0.23 ms/frame     1.8%
  resize           0.01 ms/frame     0.0%
  inference       10.86 ms/frame    84.3%   ← the only part that matters
  annotate         0.85 ms/frame     6.6%
  overlay          0.26 ms/frame     2.0%
  jpeg encode      0.62 ms/frame     4.8%
```

Two things follow. Inference dominates, so micro-optimising the CPU stages is pointless.
And **decode is 0.23 ms**, so moving `cv2.VideoCapture.read()` onto a worker thread — the
obvious "make it parallel" idea — would buy essentially nothing. Worth knowing before
building it.

---

## The six fixes

### 1. Display rate is now decoupled from inference rate

The single biggest remaining win, and the one you can see in the browser.

Inference and display are two different requirements that were one number. Tracking wants
every frame it can get; a Kalman filter's association step degrades as the gap between
observations grows. A viewer wants about 30 fps and cannot perceive more.

Sending all ~78 inferred fps meant ~78 ForwardMsgs **and ~78 browser GETs per second**,
because `st.image(bytes)` does not inline the image — it registers the payload and returns
a `/media/<content-hash>.jpg` URL that the browser fetches separately. Every frame is a
unique hash, so every frame is a guaranteed cache miss.

`_DisplayThrottle` caps only the paint rate:

| | images sent | painted fps | msgs/frame | wall | unique objects |
|---|---|---|---|---|---|
| Uncapped | 775 | 57.3 | 1.58 | 13.5 s | 122 |
| `VIDEO_DISPLAY_FPS = 30` | **265** | 23.9 | **0.88** | **11.1 s** | **122** |

66% fewer browser fetches, 44% fewer messages, 18% less wall time — and **the same 122
unique objects tracked**, because every frame still goes through the detector and the
counters. The run summary reports it honestly: `775 processed · 264 drawn (≤30 fps)`.

This is strictly better than reaching for **⏩ Skip Frames**, which buys the same transport
relief by throwing away detections. Set `VIDEO_DISPLAY_FPS = 0` to disable the cap.

### 2. Sidebar metrics are rate-limited

`_LiveMetrics.update()` now repaints at `config.METRICS_REFRESH_HZ` (default 5) instead of
every frame. The numbers stay exact — they are read from the live counters at draw time,
not accumulated in the widget — so only the paint rate changes. The final frame forces a
draw so you are left looking at true totals rather than whatever the last tick caught.

This is the fix for the paradox at the top of this document.

### 3. Streamed frames no longer leak until the video ends

`st.image(bytes)` registers every payload in Streamlit's global `MediaFileManager`. Only
the newest file at each screen coordinate stays referenced; the rest are collected by
`remove_orphaned_files()`, which runs **when a script run ends**
(`script_runner.py:906`). A playback loop *is* one script run — it does not end until the
video does, so nothing was ever collected while you watched.

Measured on `vehicle_crossing_2` (2809 frames):

| | Peak files retained | Peak bytes |
|---|---|---|
| Before | 2810 | **183.8 MB** |
| After, `_gc_media_files()` every 30 frames | 32 | **11.4 MB** |

It is worse than a plain leak. The ForwardMsg queue **coalesces messages sharing a delta
path** (`forward_msg_queue.py:96-103`), so when the loop outruns the socket, older frames
are replaced in the queue and never sent — yet their bytes were already registered and
still counted. You paid memory for frames the browser never displayed.

Cost of the fix, measured at the 30-file steady state: **2.1 ms per call, 0.07 ms/frame
amortised** — 0.6% of a frame.

> **Why 30 and not 1.** Frames are served as URLs, so collecting one the browser has not
> fetched yet would 404. Thirty frames of grace is 0.3–0.5 s at these rates, ample on
> localhost and on a normal network. If you deploy somewhere with a slow media path and see
> occasional broken frames, raise `MEDIA_GC_EVERY_N_FRAMES`; memory stays bounded either way.

### 4. Frames were **upscaled** before inference

```python
w = config.VIDEO_DISPLAY_WIDTH   # 720, applied unconditionally
```

Every clip in `videos/` except two is 640×360, so resizing *to* 720 upscaled them 1.12×.

| Input | Inference |
|---|---|
| 720 wide (upscaled) | 22.09 ms |
| 640 wide (native) | **17.26 ms** |

28% more compute and a bigger JPEG, in exchange for pixels the camera never captured —
interpolation cannot add detail a detector can use. `_display_size()` now treats
`VIDEO_DISPLAY_WIDTH` as a **ceiling**:

```python
w = min(config.VIDEO_DISPLAY_WIDTH, w_orig)
```

Downscaling still happens where it genuinely helps: `video_1.mp4` is 3840×2160.

### 5. Ultralytics logged one line per frame

`_process_frame()` called `model.track()` without `verbose=False`.

| | ms/frame |
|---|---|
| `verbose` default | 11.70 |
| `verbose=False` | **9.52** |

2.18 ms — **19% of inference** — spent formatting text nobody reads, plus a flooded terminal.

### 6. Skipped frames re-sent a picture already on screen

```python
if frame_num % skip_frames != 0:
    if last_bytes is not None:
        st_frame.image(last_bytes, width="stretch")   # ← removed
    continue
```

Re-sending the previous JPEG re-registered it in the media manager and queued another
ForwardMsg to paint an identical image. **⏩ Skip Frames** cut GPU work but bought *zero*
transport savings, which is most of what it needed to save. Skipped frames are now simply
not drawn; the last one stays up, which is both correct and free.

---

## Combined effect

Server-side, `people_crossing_1`, one video:

| | ms/frame | fps | payload |
|---|---|---|---|
| Before (720 upscale, q90, verbose) | 14.02 | 71.3 | 84.9 KB/frame |
| After (native 640, q75, quiet) | **10.98** | **91.1** | **49.6 KB/frame** |

1.28× on compute and 42% less on the wire (83 MB → 48 MB per 1000 frames), on top of the
transport and memory fixes above — which is where the perceived slowness actually lived.

---

## Why a Kafka pipeline felt faster than this

Because it was structurally different in the one way that matters.

A Streamlit playback loop is **serial on one thread**: decode → infer → annotate → encode →
register media → enqueue → repeat. Nothing overlaps, so every millisecond of transport
overhead is a millisecond the GPU sits idle.

A producer/consumer split over Kafka gives decode, inference and delivery their own threads
or processes. They pipeline. The broker hop adds *latency* per frame but raises
*throughput*, and throughput is what you perceive as smooth playback. Adding brokers and
consumers made it faster because it stopped one thread from doing everything in sequence.

The fixes above do not change that architecture; they cut the serial work down to roughly
inference plus a JPEG encode. Note from the profile that the classic next step —
decoding on a worker thread — is **not** worth it here: decode is 0.23 ms of a 12.89 ms
frame. If you need to go further, the thing to move off the script thread is inference
itself.

---

## Frame pipeline

```
cv2.VideoCapture.read()          CPU   decode                       0.23 ms
_display_size() + cv2.resize()   CPU   downscale only, never up      0.01 ms
model.track(verbose=False)       GPU   inference + tracking         10.86 ms
_annotate_with_ids()             CPU   boxes, labels, track IDs      0.85 ms
_draw_overlay()                  CPU   local/global/quality stats    0.26 ms
cv2.imencode(".jpg", q=75)       CPU   only for painted frames       0.62 ms
st_frame.image(bytes)            NET   1 ForwardMsg + 1 GET /media/<hash>.jpg
_gc_media_files()                      every 30 frames               2.1 ms
```

---

## Configuration knobs

| Setting | File | Default | Effect |
|---|---|---|---|
| `VIDEO_DISPLAY_FPS` | `config.py` | `30.0` | Paint-rate ceiling; `0` disables. Does **not** affect detection |
| `VIDEO_DISPLAY_WIDTH` | `config.py` | `720` | **Ceiling** on inference width; never upscales |
| `VIDEO_JPEG_QUALITY` | `config.py` | `75` | 82 KB/frame at q90 → 49 KB at q75, no visible loss |
| `METRICS_REFRESH_HZ` | `config.py` | `5.0` | Sidebar repaint rate; each repaint is ~7 ForwardMsgs |
| `MEDIA_GC_EVERY_N_FRAMES` | `config.py` | `30` | Frame-JPEG collection interval; also the 404 grace window |
| `DEFAULT_SKIP_FRAMES` | `config.py` | `1` | Infer every Nth frame — GPU only, and it costs accuracy |
| `MIN_TRACK_HITS` | `config.py` | `5` | Frames before a track counts as real |

### Tuning order

1. **Model size first.** `yolo26n` → `yolo26l` is several times the latency. Nano is usually enough.
2. **`🎯 Limit to classes`** — the cheapest win in the app. Dropped classes cost no annotation,
   no counting, and with ReID no crop and no embedding.
3. **`VIDEO_DISPLAY_FPS`** if the browser still struggles. Costs you nothing but smoothness.
4. **`⏩ Skip Frames`** only if the GPU is genuinely the constraint — check the `FPS` metric
   against the clip's native rate first. See §9.2 of
   [`yolo26_playground.ipynb`](yolo26_playground.ipynb): skipping distorts unique-object
   counts in both directions.
5. **ReID last.** Roughly two-thirds of your FPS. Prefer skip=2 *with* ReID over skip=1 without.

---

## Reproducing these numbers

Streamlit's `AppTest` harness plus two hooks:

- `ForwardMsgQueue._before_enqueue_msg` — a class-level hook counting every message the
  server pushes, by element type. This produced the tables above.
- Wrapping `DeltaGenerator.image` to sample
  `get_instance().media_file_mgr._storage._files_by_id` on every painted frame. This
  produced the 183.8 MB figure.

Run each configuration in **its own process**. `config` values read at import time — such as
a function's default argument — will not pick up a change made afterwards, which is a good
way to measure nothing and believe it.

Both hooks are headless, so they measure the server honestly and say nothing about browser
render time.
