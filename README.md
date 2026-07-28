# Mercury

**Embedded speech recognition in pure Rust.** Reimagined from Whisper.cpp with no Python runtime, no C/C++ by default. Built for edge, browser, embedded applications that require low memory and compute requirements.

Mercury is the voice component of [FFai](https://github.com/Remade-With-Rust/FFai),
published as a standalone crate so you can use ASR without the rest of the
toolkit. Part of [Remade With Rust](https://github.com/Remade-With-Rust).

```toml
[dependencies]
ffai-mercury = "0.1"
```

```rust
use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_mercury::asr::WhisperCandle;

let engine = WhisperCandle::new();                  // whisper-tiny-en
let audio  = ffai_media::load_audio("talk.wav")?;   // 16 kHz mono
let transcript = engine.transcribe(&audio, &AsrOptions::default())?;

for segment in &transcript.segments {
    println!("[{:.2}–{:.2}] {}", segment.start, segment.end, segment.value);
}
```

Weights are fetched into a local cache from hash-verified manifests on first
use — never vendored, and each model's own licence is surfaced at selection
time.

## Where it stands

Measured against **whisper.cpp** (C++/ggml) on two hash-pinned **134-clip**
LibriSpeech holdouts, matched greedy decoding, CPU only, tiny.en:

| Corpus | Implementation | WER % | CER % | ×realtime (warm) |
|---|---|---:|---:|---:|
| test-clean | **Mercury** (Rust) | 7.77 | 3.25 | 32.1–32.8 |
| test-clean | whisper.cpp | **7.58** | **2.87** | **35.7–36.6** |
| test-other | **Mercury** (Rust) | **16.79** | **8.34** | 26.7 |
| test-other | whisper.cpp | 16.82 | 8.41 | **29.5** |

**Ahead of whisper.cpp on noisy speech, 0.19 pp behind on clean speech, and
~1.12× behind on throughput.** The engine is honestly labelled
`experimental`, not `stable`: the quality gate passes on both corpora, the
speed gate fails, and the footprint gate is not yet instrumented — and a
skipped gate is never a pass, so the four-gate verdict is *not claimable yet*.

Worth knowing what the bar is. whisper.cpp is not a naive baseline: it runs
flash attention on by default, an OpenBLAS backend, runtime ISA dispatch to an
AVX-VNNI build, blocked weight repacking, and f16 weights. Toggling its own
`-nfa` flag prices that fused attention at **1.65×** — and against its
*unfused* encoder Mercury is **1.38× faster**.

Every number above traces to a line in the
[claims ledger](https://github.com/Remade-With-Rust/FFai/blob/master/bench/ledger.jsonl),
with the full methodology — including every reverted experiment — in
[docs/whys/](https://github.com/Remade-With-Rust/FFai/tree/master/docs/whys).

## What works today

- Whisper `tiny.en` and `base.en`, greedy decoding with the full logit-filter
  grammar (suppression lists, timestamp rules, temperature fallback).
- Segment-level timestamps, 30 s windowing with context carry-over.
- int8 and f16 decoder variants (`whisper-candle-q8_0`) — memory, not speed.

## What does not, yet

- **TTS.** The `TtsEngine` trait exists; Kokoro-82M is the first target.
- **Beam search**, all model sizes above `base`, streaming API.
- **Word-level timestamps and diarization** — the WhisperX layer, planned as
  flags over any registered engine rather than a fork.

See the [FFai roadmap](https://github.com/Remade-With-Rust/FFai/blob/master/ROADMAP.md)
for sequencing.

## Related crates

| Crate | What it is |
|---|---|
| [`ffai-mercury`](https://crates.io/crates/ffai-mercury) | this — ASR (and TTS, when it lands) |
| [`ffai-core`](https://crates.io/crates/ffai-core) | engine traits, shared types, registry (candle is the tensor spine) |
| [`ffai-models`](https://crates.io/crates/ffai-models) | weight manifests, hash-verified cache, licence surfacing |
| [`ffai-media`](https://crates.io/crates/ffai-media) | audio ingest/egress |
| [`ffai-bench`](https://crates.io/crates/ffai-bench) | the analyzer: four-gate verdicts, best-of-N timing, pinned corpora, claims ledger |

## Licence

MIT OR Apache-2.0 (code). Model weights carry their own licences — surfaced at
selection time, and often more restrictive than this crate's.
