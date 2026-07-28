# Mercury

**Embedded speech recognition in pure Rust.** Reimagined from Whisper.cpp & WhisperX with no Python runtime, no C/C++ by default. Built for edge, browser, embedded applications that require low memory and compute requirements.

Mercury is the voice component of [FFai](https://github.com/Remade-With-Rust/FFai),
published as a standalone crate so you can use ASR without the rest of the
toolkit. Part of [Remade With Rust](https://github.com/Remade-With-Rust).

```toml
[dependencies]
ffai-mercury = "0.2"
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

Per-word timings, when you need them:

```rust
let opts = AsrOptions { word_timestamps: true, ..Default::default() };
let transcript = engine.transcribe(&audio, &opts)?;

for word in transcript.words.iter().flatten() {
    println!("{:6.2}–{:6.2}  {}", word.start, word.end, word.value);
}
```

Weights are fetched into a local cache from hash-verified manifests on first
use — never vendored, and each model's own licence is surfaced at selection
time.

## Where it stands

Measured against **whisper.cpp** (C++/ggml) on two hash-pinned **134-clip**
LibriSpeech holdouts, matched greedy decoding, CPU only, tiny.en:

| Corpus | Implementation | WER % | CER % | ×realtime (warm) | steady MiB |
|---|---|---:|---:|---:|---:|
| test-clean | **Mercury** (Rust) | **6.79** | **2.74** | 32.9 | **183** |
| test-clean | whisper.cpp | 7.58 | 2.87 | **33.2–36.6** | 194 |
| test-other | **Mercury** (Rust) | **16.43** | **8.07** | 26.7 | **167** |
| test-other | whisper.cpp | 16.82 | 8.41 | **29.0–29.5** | 192 |

Ahead on WER and CER on both corpora, ahead on memory, 1.01–1.09× on speed.

**And here is the asterisk, because it belongs next to the numbers rather
than at the bottom of a page.** Part of that quality margin comes from speech
segmentation being on by default, which whisper.cpp does not do — and
segmentation is *not* a quality mechanism. Turning it off gives 7.99 / 16.79.
That looks like a 1.20 pp win and is not one: decomposed per clip across 400
clips it is **38 improved, 38 worsened — a sign test of z = 0.00**, and the
correlation between silence removed and WER gained is **−0.09**, the opposite
sign to the mechanism originally proposed for it. It shifts where speech sits
inside Whisper's fixed 30 s context and re-rolls the decode on about a fifth
of clips, half each way; the aggregate moved because WER is dominated by a
handful of high-delta clips. **Do not expect this margin to transfer to your
audio.** The full descent, including the two occasions the wrong conclusion
was drawn before the distribution was examined, is in
[whys/vad-quality.md](https://github.com/Remade-With-Rust/FFai/blob/master/docs/whys/vad-quality.md).

Segmentation ships for its **speed**, which *is* a mechanism rather than a
delta: 2.2–4.2× on audio with trailing silence at a byte-identical transcript,
and silence producing an empty transcript with no encoder pass at all.

The engine is honestly labelled `experimental`, not `stable`. The
correctness and footprint gates pass; the speed gate does not; and the
harness's quality gate compares against the best of *all* references —
`openai-whisper-base`, a 74M beam-search model — rather than like for like.
Against **matched** references (tiny, greedy) Mercury's 6.79 % is first, ahead
of faster-whisper-tiny-greedy (7.04 %), openai-whisper-tiny-greedy (7.41 %)
and whisper.cpp (7.58 %). A skipped or mismatched gate is never a pass, so the
four-gate verdict remains *not claimable yet*.

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
- **Voice activity detection, on by default** — speech-shaped windows instead
  of a fixed 30 s grid. `AsrOptions { vad: false, .. }` restores the grid.
- **Word-level timestamps** (`word_timestamps: true`) via CTC forced
  alignment against a wav2vec2 acoustic model, ported to candle.
- A **no-speech gate**: silence yields an empty transcript rather than the
  `you` that Whisper hallucinates into it.
- Non-speech events annotated (`[Laughs]`, `(coughs)`) as whisper.cpp does;
  `DecodeConfig::suppress_non_speech` restores openai-whisper's behaviour.
- Segment-level timestamps, SRT / WebVTT (with inline word timing tags) /
  JSON output.
- int8 and f16 decoder variants (`whisper-candle-q8_0`) — memory, not speed.

## What does not, yet

- **TTS.** The `TtsEngine` trait exists; Kokoro-82M is the first target.
- **Diarization** — speaker labels, the last of the WhisperX layer.
- **Beam search**, all model sizes above `base`, streaming API.
- **Word timestamps are English-only** and the alignment model is verified on
  one clip's output, not yet against the reference implementation's emissions
  over a corpus. Treat the timings as good, not as gospel.

See the [FFai roadmap](https://github.com/Remade-With-Rust/FFai/blob/master/ROADMAP.md)
and the [Mercury-X plan](https://github.com/Remade-With-Rust/FFai/blob/master/docs/mercury-X-mission.md)
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
