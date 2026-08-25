# Native-video chronology validation

Date: 2026-08-25  
Platform: NVIDIA Jetson AGX Thor  
Runtime: FlashRT + Cosmos3 Edge, BF16  
Transport: ROS 2 image topic -> sampled temporal window -> versioned IPC -> native-video preprocessing

## Purpose

This experiment tests whether native-video inference is sensitive to frame chronology rather than merely to the set of visual content supplied. The same captured frames are replayed under controlled order manipulations while frame count, timestamp schedule, prompt, and runtime path are held constant.

## Captured window

- 8 frames
- relative timestamps: `0.000, 0.267, 0.534, 0.801, 1.067, 1.334, 1.601, 1.868 s`
- total span: **1.868 s**
- capture motion score: **0.253653**
- forward/reverse/shuffled/static variants all used `flashrt_cosmos3_native_video`

## Results

| Variant | Frame order | Inference | Observation |
| --- | --- | ---: | --- |
| Forward | `0,1,2,3,4,5,6,7` | **6.192 s** | Person walking **right to left**, moving closer; camera reported panning left |
| Reverse | `7,6,5,4,3,2,1,0` | **6.139 s** | Person walking **left to right** |
| Shuffled | `0,7,1,6,2,5,3,4` | **6.137 s** | Model still produced a coherent walking narrative, including a right-to-left summary |
| Static terminal | `7,7,7,7,7,7,7,7` | **6.157 s** | Structured `CHANGES` reported **none**, while the free-form summary still described walking left to right |
| Terminal only | `7` | **3.854 s** | Single-image diagnostic inferred a walking/forward action from appearance; this is not temporal displacement evidence |

## Interpretation

1. **Chronology sensitivity is demonstrated for this sequence.** Reversing the exact same eight images while preserving the timestamp schedule reversed the inferred lateral motion direction.
2. **The repeated-frame native-video control correctly reported no temporal change in the structured change field.** The video path does not automatically invent displacement solely because it receives multiple frames.
3. **Temporal coherence rejection is not established.** The shuffled sequence was temporally incoherent but still produced a plausible motion narrative.
4. **Free-form summaries are less reliable than structured fields for evaluation.** The repeated-frame control contained a contradiction between `CHANGES: none` and the summary.
5. **Single-frame action inference should not be scored as temporal motion.** A walking pose can reasonably suggest an action state without proving displacement over time.

## Current conclusion

Native Cosmos3 video inference is a valid temporal baseline: it responds to chronology and preserves exact timestamp semantics. General temporal reliability remains an open evaluation question and should be measured across multiple motion-rich windows and multiple video-capable models using the same saved captures and controls.

## Recommended next benchmark

Use a shared model-agnostic corpus of saved temporal windows and score each model on forward direction accuracy, reverse consistency, static false-change rate, shuffled-sequence rejection, structured-output compliance, and latency.
