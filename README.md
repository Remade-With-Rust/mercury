# Mercury

**Embedded speech recognition in pure Rust.** Reimagined from Whisper.cpp & WhisperX with no Python runtime, no C/C++ by default. Built for streaming AI-multi media from edge, browser, and embedded applications with low memory and compute requirements.

Mercury is the voice component of [FFai](https://github.com/Remade-With-Rust/FFai),
published as a standalone crate so you can use ASR without the rest of the
toolkit. Part of [Remade With Rust](https://github.com/Remade-With-Rust).

```toml
[dependencies]
ffai-mercury = "0.4"
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

The WhisperX layer, when you need it:

```rust
let opts = AsrOptions {
    word_timestamps: true,          // per-word times, CTC forced alignment
    diarize: true,                  // speaker turns
    ..Default::default()            // speech segmentation is already on
};
let transcript = engine.transcribe(&audio, &opts)?;

for word in transcript.words.iter().flatten() {
    println!("{:6.2}–{:6.2}  {}", word.start, word.end, word.value);
}
for turn in transcript.speakers.iter().flatten() {
    println!("{:6.2}–{:6.2}  {}", turn.start, turn.end, turn.value);  // SPEAKER_00…
}
```

Or from the CLI:

```sh
ffai asr -i meeting.wav -o meeting.json --word-timestamps --diarize
```

| Stage | Model fetched | Gate |
|---|---|---|
| segmentation *(default on)* | none — energy VAD | silence corpus, **8/8 empty** |
| `word_timestamps` | wav2vec2-base-960h, Apache-2.0 | containment **100 %**, 1105 words |
| `diarize` | ECAPA-TDNN, Apache-2.0 | **DER 4.21 %** |
| `persist_speakers` | — | streaming **DER 5.68 %** |

The opt-in stages are lazy: without the flag their models are not fetched, not read, and not resident.

### Streaming: labels that survive the next chunk

Diarization labels are, by convention, arbitrary names for clusters *within one call* — `SPEAKER_00` in two separate calls need not be the same person. Fine for a file; useless for a live stream, where the same voice gets renamed every chunk.

```rust
let opts = AsrOptions { diarize: true, persist_speakers: true, ..Default::default() };

for chunk in microphone_chunks {
    let t = engine.transcribe(&chunk, &opts)?;   // SPEAKER_00 is the same
    // …                                          // person in every chunk
}
engine.reset_speakers();                          // new recording, new people
```

Conversations fed as 8 s chunks, DER scored over the whole concatenated timeline: **53.58 % without it, 5.68 % with it.** Matching is deliberately stricter than in-call clustering — a registry merge is permanent, and two people who share a centroid stay merged for the rest of the session.

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

## See it run

The [FFai repo](https://github.com/Remade-With-Rust/FFAI) carries a live
side-by-side demo: speak into a microphone and read Mercury and whisper.cpp
transcribing the *same* audio, in real time, in two panes.

```sh
git clone https://github.com/Remade-With-Rust/FFAI && cd FFAI
cargo run --release -p ffai-demo     # then open http://127.0.0.1:8787
```

Three things are visible there that a WER table cannot show. Silence produces
**nothing** from Mercury while whisper.cpp prints `[BLANK_AUDIO]`. Speaker
labels appear inline and **hold steady across chunks** — stop talking, let
someone else speak, come back, and your original label returns. And on
ordinary speech the two panes agree word for word, which is the point.

## What works today

- Whisper `tiny.en` and `base.en`, greedy decoding with the full logit-filter
  grammar (suppression lists, timestamp rules, temperature fallback).
- **The complete WhisperX layer** — segmentation, word-level timestamps by
  CTC forced alignment, and speaker diarization — all in candle, all behind
  ungated Apache-2.0 weights.
- A **no-speech gate**: silence yields an empty transcript rather than the
  `you` that Whisper hallucinates into it.
- Non-speech events annotated (`[Laughs]`, `(coughs)`) as whisper.cpp does;
  `DecodeConfig::suppress_non_speech` restores openai-whisper's behaviour.
- SRT / WebVTT (inline word-timing tags) / JSON carrying words and speakers.
- int8 and f16 decoder variants (`whisper-candle-q8_0`) — memory, not speed.

## Three things measurement taught us

**Segmentation is on for speed, not quality.** 2.2–4.2× on audio with trailing
silence at a byte-identical transcript. It also moves corpus WER, and that is
*not* a quality win — 38 improved / 38 worsened across 400 clips, sign test
z = 0.00. Set `vad: false` for the fixed 30 s grid.

**`max_speakers` is not the safe option.** Blind clustering scores **4.21 %
DER** against **5.00 %** with the true speaker count supplied. Forcing a count
forces a merge, and a bad merge attributes one speaker's words to another.
Supply it when it is certain, not as insurance.

**Licences shaped this.** WhisperX's diarization uses pyannote weights that
are MIT-licensed *and gated* — permission granted, access behind a browser
click. Mercury uses SpeechBrain's ECAPA-TDNN instead: Apache-2.0 and ungated.
Every model Mercury fetches is fetchable without an account.

## What does not, yet

- **TTS.** The `TtsEngine` trait exists; Kokoro-82M is the first target.
- **Beam search**, all model sizes above `base`, streaming API.
- **Word timestamps are English-only.** The alignment model is per-language.
- **Word-level timing is gated at utterance granularity**, not milliseconds —
  words are proven to land in the right utterance across a 185 s multi-window
  file, not that a boundary is accurate to 50 ms. Finer needs a reference
  aligner.
- **The diarization corpus has no speaker overlap** and no natural
  turn-taking. It gates *regression*, not *readiness*: a system can score
  4.21 % there and do worse on a real meeting.

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
