# Mercury

**Embedded speech recognition *and synthesis* in pure Rust.** Reimagined from Whisper.cpp & WhisperX with no Python runtime, no C/C++ by default — and now text-to-speech reimagined from Piper, with no GPL. Built for streaming AI multi-media from edge, browser, and embedded applications with low memory and compute requirements.

Mercury is the voice component of [FFai](https://github.com/Remade-With-Rust/FFai),
published as a standalone crate so you can use speech without the rest of the
toolkit. Part of [Remade With Rust](https://github.com/Remade-With-Rust).

```toml
[dependencies]
ffai-mercury = "0.4"
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
[Piper](https://github.com/OHF-Voice/piper1-gpl) runs, converted locally from
the voice's own `.onnx`.

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

That boundary is measured, not asserted. The **substitution gate** feeds
*our* phonemes through *Piper's own runtime* and scores the resulting audio
against espeak's phonemes through the same runtime — synthesis held constant,
so the difference prices the phonemizer and nothing else. It passes inside the
5 % relative band.

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

Read the WER column as line-ball rather than a win: ahead by 0.31 pp on clean,
behind by 0.07 pp on noisy.

**And here is the asterisk, because it belongs next to the numbers rather than
at the bottom of a page.** Speech segmentation is on by default, which
whisper.cpp does not do — and segmentation is *not* a quality mechanism.
Decomposed per clip across 400 clips its effect is **38 improved, 38 worsened
— a sign test of z = 0.00**, and the correlation between silence removed and
WER gained is **−0.09**, the opposite sign to the mechanism originally
proposed for it. It shifts where speech sits inside Whisper's fixed context and
re-rolls the decode on about a fifth of clips, half each way. **Do not expect
a segmentation margin to transfer to your audio.** The full descent, including
the two occasions the wrong conclusion was drawn before the distribution was
examined, is in
[whys/vad-quality.md](https://github.com/Remade-With-Rust/FFai/blob/master/docs/whys/vad-quality.md).

Segmentation ships for its **speed**, which *is* a mechanism rather than a
delta: 2.2–4.2× on audio with trailing silence at a byte-identical transcript,
and silence producing an empty transcript with no encoder pass at all.

Worth knowing what the bar is. whisper.cpp is not a naive baseline: it runs
flash attention on by default, an OpenBLAS backend, runtime ISA dispatch to an
AVX-VNNI build, blocked weight repacking, and f16 weights.

### Speech synthesis

Measured against **piper1-gpl** (Python + onnxruntime + espeak-ng) on a
hash-pinned 200-sentence Harvard corpus, 134 holdout, same voice
(`en_US-lessac-medium`), same knob values:

| | round-trip WER % | ×realtime (warm) | steady MiB | load s |
|---|---:|---:|---:|---:|
| **Mercury** (Rust) | **5.91**, byte-stable | 19–20 | **172–208** | **0.26–0.35** |
| piper1-gpl | 4.8–6.5, one draw per run | **25–32** | 217–240 | 1.8–2.6 |

**Correctness is oracle-exact against Piper's own runtime.** At zero noise
both implementations are deterministic functions of the same phoneme ids, so
every stage is pinned against onnxruntime's own intermediates: text encoder to
**4e-6**, per-phoneme durations **integer-exact**, end-to-end waveform to
**3e-5**.

**Quality is parity, and the instrument is why that is the honest word.**
Round-trip WER means synthesize the corpus, transcribe it with a *frozen
third-party* ASR — whisper.cpp, pinned, never Mercury's own engine, because
self-grading is not measurement — and score the transcript against the input
text. Mercury reads **5.91 % on every run**. Piper samples its noise inside
the ONNX graph with no seed control, so it cannot repeat a number: across
runs it has drawn anywhere in **4.8–6.5 %**, and the harness therefore scores
it as the mean of independent draws with the range recorded in the ledger
line. Our number sits inside its distribution. That supports *parity through
this instrument* — not superiority, and we will not write superiority until an
instrument can carry it.

**Determinism is a capability, not a detail.** Same text, same seed,
byte-identical WAV — verified at both the library and the file-hash level.
Piper structurally cannot offer it. Testing, caching, and byte-identical A/B
gating all hang off that property, and it is why the WER column above has one
number instead of a range.

**Speed is behind, closing, and not claimable.** Synthesis went from 3.2×
realtime at bring-up to 19–20× warm across five profiled campaigns —
cache-blocked and quad-packed AVX2 convolution kernels, phase-decomposed
upsamplers, a flat decoder that never round-trips through tensors, a
GEMM-shaped coupling flow with vectorized-exp gates. Nine attempts were
measured, refuted, and reverted along the way; all nine are in the plan beside
the wins. Function by function against Piper's own runtime, Mercury's
**upsamplers and duration predictor are ~1.9× faster** and the text encoder is
at parity — the remaining gap lives in two convolution kernels whose targets
are measured, not guessed. The speed gate reads FAIL on every fair ledger
line, and three machine-compromised lines (two that flattered Piper, one that
flattered Mercury) are explicitly disowned in the plan with reading
instructions.

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
