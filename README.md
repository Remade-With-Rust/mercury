# Mercury

**Embedded speech recognition (ASR) *and* text-to-speech (TTS) in pure Rust.** Reimagined from Whisper.cpp & WhisperX with no Python runtime, no C/C++ by default — and now text-to-speech reimagined from Piper, with no GPL. Built for streaming AI multi-media from edge, browser, and embedded applications with low memory and compute requirements.

Mercury is the Roman god of language and messages

Mercury is the voice component of [FFai](https://github.com/Remade-With-Rust/FFai),
published as a standalone crate so you can use speech without the rest of the
toolkit. Part of [Remade With Rust](https://github.com/Remade-With-Rust).

```toml
[dependencies]
ffai-mercury = "0.6"
```

## Speech → text

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

## Text → speech

```rust
use ffai_core::engine::{TtsEngine, TtsOptions};
use ffai_mercury::tts::PiperCandle;

let engine = PiperCandle::new();
let audio  = engine.synthesize("The birch canoe slid on the smooth planks.",
                               &TtsOptions::default())?;
ffai_media::save_wav("out.wav".as_ref(), &audio)?;
```

```sh
ffai tts "Hello from Mercury." -o hello.wav
ffai tts -o out.wav --seed 42 "Same seed, same bytes."
```

`piper-candle` is the full **VITS** stack on candle — text encoder with
relative-position attention, stochastic duration predictor with spline flows,
residual coupling flow, HiFi-GAN vocoder — running the **same voice files**
[Piper](https://github.com/OHF-Voice/piper1-gpl) runs — fetched from the
public, ungated `rhasspy/piper-voices` and read straight from ONNX by our own
pure-Rust reader. No conversion step, no Python, no ONNX runtime:
`ffai models --fetch piper-vits-lessac-medium` is the whole setup.

| Option | Effect |
|---|---|
| `speed` | playback rate; 1.0 is the voice's own timing |
| `noise_scale` / `noise_w` | acoustic and duration variation; `0.0` = fully deterministic audio |
| `seed` | noise seed — same text + same seed gives a **byte-identical WAV** |
| `sentence_silence_s` | gap inserted between sentences of long-form input |

Long-form input is segmented into sentences, synthesized per sentence, and
joined, so `ffai tts` on a paragraph just works.

### The phonemizer is ours, and that is a licensing decision

Piper is GPL-3.0 because it embeds espeak-ng. Mercury's grapheme-to-phoneme
stage is a clean-room pure-Rust implementation over CMUdict (BSD-2-Clause)
that emits espeak-compatible IPA; espeak-ng participates **only as an
out-of-process test oracle** over pinned corpora, and nothing GPL is linked,
vendored, or shipped. The honest cost, stated rather than buried: **en-US
only** for now, where Piper covers 40+ languages.

## Where it stands

### Speech recognition

Measured against **whisper.cpp** (C++/ggml) on two hash-pinned **134-clip**
LibriSpeech holdouts, matched greedy decoding, CPU only, tiny.en:

| Corpus | Implementation | WER % | ×realtime (warm) | steady MiB |
|---|---|---:|---:|---:|
| test-clean | **Mercury** (Rust) | **7.27** | **27.8** | **179** |
| test-clean | whisper.cpp | 7.58 | 25.9 | 195 |
| test-other | **Mercury** (Rust) | 16.89 | **33.3** | **163** |
| test-other | whisper.cpp | **16.82** | 19.6 | 194 |

**All four gates pass on both holdouts** — correctness, quality, speed, and
footprint. Speed had failed every previous ledger line; **adaptive encoder
context** closed it, encoding each window at a context sized to the audio
actually present rather than always 30 s, with guards that escalate a suspect
decode back to the full context. Function by function, Mercury is now ahead of
whisper.cpp on **every** stage: encode ~2.0×, decode 1.1–1.2×, mel 1.4×,
sampling 1.7–2.0×.

Worth knowing what the bar is. whisper.cpp is not a naive baseline: it runs
flash attention on by default, an OpenBLAS backend, runtime ISA dispatch to an
AVX-VNNI build, blocked weight repacking, and f16 weights.

### Speech synthesis

Measured against **piper1-gpl** (Python + onnxruntime + espeak-ng) on a
hash-pinned 200-sentence Harvard corpus, 134 holdout, same voice
(`en_US-lessac-medium`), same knob values:

| | round-trip WER % | ×realtime (warm) | steady MiB | load s |
|---|---:|---:|---:|---:|
| **Mercury** (Rust) | **5.49** (4.99–6.10 across seeds) | 19.3–21.2 | 214 | **0.64** |
| piper1-gpl | **5.27** (4.8–6.5 across runs) | 11.2–23.0 | 217 | 6.69 |

Both columns are ranges on purpose — see *"one draw is not a number"* below.
The single figure that is measured on **both engines simultaneously**, and so
survives the variance: **1.58× faster wall-clock while using 5 % less total
CPU.**

**Correctness is oracle-exact against Piper's own runtime.** At zero noise
both implementations are deterministic functions of the same phoneme ids, so
every stage is pinned against onnxruntime's own intermediates: text encoder to
**4e-6**, per-phoneme durations **integer-exact**, end-to-end waveform to
**3e-5**. The pure-Rust ONNX reader that replaced the Python converter is
**byte-identical** to it — 350 tensors, 15.65 M floats, 132 convolution
geometries and the audio itself, all exact — so it inherits that oracle by
construction rather than by re-argument.

**Quality is parity**
Round-trip WER means synthesize the corpus, transcribe it with a *frozen
third-party* ASR — whisper.cpp, pinned, never Mercury's own engine, because
self-grading is not measurement — and score the transcript against the input
text. Mercury reads **5.49 %** against piper's **5.27 %** on the same holdout,
same judge, same run.

**Speed: ahead, on a reference that will not hold still.** Recent ledger lines
read **19.3–21.2× realtime warm against piper's 11.2–23.0×**, and **load 0.64 s
against 6.69 s**. But piper's own throughput spans **4.5×–29× across the
ledger's TTS lines** on this machine — wider than any difference between the
two engines — so single-run ratios are not a claim, and the four gates have
flipped pass/fail on *its* variance rather than ours. What does survive is the
simultaneous measurement: **1.58× faster wall at 5 % less CPU**, and footprint
at parity.

Every number above traces to a line in the
[claims ledger](https://github.com/Remade-With-Rust/FFai/blob/master/bench/ledger.jsonl).
Full campaign histories, every reverted experiment included:
[ASR](https://github.com/Remade-With-Rust/FFai/blob/master/docs/finished/mercury-mission-plan.md)
· [TTS](https://github.com/Remade-With-Rust/FFai/blob/master/docs/mercury-tts-mission.md)
· [the whys](https://github.com/Remade-With-Rust/FFai/tree/master/docs/whys).

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

The **Speak** tab does the same for synthesis: type anything, hear it back, and
see the phonemes our G2P produced, where the sentence split landed, and a
*Speak twice* button that renders the same input twice and compares a SHA-256
of the samples — determinism made checkable rather than claimed.

## What works today

**Recognition**

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

**Synthesis**

- The full VITS stack on candle, running Piper's own voice files, gated
  stage-by-stage against Piper's runtime.
- A pure-Rust en-US phonemizer over CMUdict — no espeak-ng, no GPL, no gated
  weights, nothing to click through.
- **Deterministic output** under a seed, plus `speed`, `noise_scale`,
  `noise_w`, and `sentence_silence_s`.
- Long-form text: sentence segmentation, per-sentence synthesis, seamless
  concatenation.

## Three things measurement taught us

**Segmentation is on for speed, not quality.** 2.2–4.2× on audio with trailing
silence at a byte-identical transcript. It also moves corpus WER, and that is
*not* a quality win — 38 improved / 38 worsened across 400 clips, sign test
z = 0.00. Set `vad: false` for the fixed 30 s grid.

**`max_speakers` is not the safe option.** Blind clustering scores **4.21 %
DER** against **5.00 %** with the true speaker count supplied. Forcing a count
forces a merge, and a bad merge attributes one speaker's words to another.
Supply it when it is certain, not as insurance.

**Licences shaped the architecture twice.** WhisperX's diarization uses
pyannote weights that are MIT-licensed *and gated* — permission granted,
access behind a browser click — so Mercury uses SpeechBrain's ECAPA-TDNN
instead: Apache-2.0 and ungated. Piper is GPL because it embeds espeak-ng, so
Mercury's phonemizer is clean-room Rust over a BSD lexicon. In both cases the
licence did not merely change the paperwork; it changed what got built. Every
model Mercury fetches is fetchable without an account.

## What does not, yet

- **TTS is en-US and single-voice.** The G2P is per-language and the voice
  tier sweep is unbuilt; the model side already supports the whole
  `rhasspy/piper-voices` family.
- **TTS synthesis speed** trails Piper on a fair line, with the two remaining
  kernels named and measured.
- **Long-form ASR still trails whisper.cpp** by 0.42 pp on a 465 s corpus,
  from one diagnosed failure class — an utterance absorbed between two
  abutting segments — with word-level coverage named as the fix.
- **Beam search**, all ASR model sizes above `base`, multilingual ASR and
  language detection.
- **Word timestamps are English-only.** The alignment model is per-language.

See the [FFai roadmap](https://github.com/Remade-With-Rust/FFai/blob/master/ROADMAP.md),
the [Mercury-X plan](https://github.com/Remade-With-Rust/FFai/blob/master/docs/finished/mercury-X-mission.md)
and the [ASR gap inventory](https://github.com/Remade-With-Rust/FFai/blob/master/docs/mercury-asr-todo.md)
for sequencing.

## Related crates

| Crate | What it is |
|---|---|
| [`ffai-mercury`](https://crates.io/crates/ffai-mercury) | this — ASR and TTS |
| [`ffai-core`](https://crates.io/crates/ffai-core) | engine traits, shared types, registry (candle is the tensor spine) |
| [`ffai-models`](https://crates.io/crates/ffai-models) | weight manifests, hash-verified cache, licence surfacing |
| [`ffai-media`](https://crates.io/crates/ffai-media) | audio ingest/egress |
| [`ffai-bench`](https://crates.io/crates/ffai-bench) | the analyzer: four-gate verdicts, best-of-N timing, pinned corpora, claims ledger |

## Licence

MIT OR Apache-2.0 (code). Model weights carry their own licences — surfaced at
selection time, and often more restrictive than this crate's. The Piper voices
are governed by their own MODEL_CARDs in
[`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices); check
one before commercial use.

---

<!-- HARDENING-TABLE:BEGIN generated by use-protection-please — edit docs/plans/use-protection-please.md, not this block -->
## Hardening status

**Tier** critical-path · **Audited** 2026-08-15 (survey) · **v1.0.0 gates** 1/16 · [Full checklist](https://github.com/Remade-With-Rust/FFAI/blob/hardening/mercury-audit/crates/ffai-mercury/docs/plans/use-protection-please.md)

`█░░░░░░░░░░░░░░░░░░░` **6%** &nbsp;·&nbsp; 2 Completed · 0 Scheduled · 33 Incomplete · 6 N/A

| Phase | ✅ Completed | 🗓 Scheduled | ⬜ Incomplete | · N/A |
|---|--:|--:|--:|--:|
| 0 — Threat modeling | 0 | 0 | 2 | 0 |
| 1 — Toolchain | 0 | 0 | 4 | 0 |
| 2 — Supply chain | 2 | 0 | 6 | 0 |
| 3 — Code level | 0 | 0 | 7 | 0 |
| 4 — Static analysis | 0 | 0 | 1 | 0 |
| 5 — Dynamic analysis | 0 | 0 | 3 | 0 |
| 6 — Fuzzing and properties | 0 | 0 | 4 | 0 |
| 7 — Formal verification | 0 | 0 | 1 | 0 |
| 8 — Build and binary | 0 | 0 | 0 | 2 |
| 9 — Runtime privilege | 0 | 0 | 0 | 1 |
| 10 — Cryptography | 0 | 0 | 0 | 3 |
| 11 — CI/CD, release, and operations | 0 | 0 | 5 | 0 |
| **Total** | **2** | **0** | **33** | **6** |
<!-- HARDENING-TABLE:END -->
