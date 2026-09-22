import sys
import json
import whisper
import librosa
import numpy as np

def get_transcript(audio_path, model_size="base"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, word_timestamps=True)
    words = []
    for segment in result["segments"]:
        for w in segment.get("words", []):
            words.append({
                "word": w["word"].strip(),
                "start": round(w["start"], 3),
                "end": round(w["end"], 3)
            })
    return words

def get_rhythm(y, sr):
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return {
        "bpm": round(float(tempo), 2),
        "beat_times": [round(float(t), 3) for t in beat_times],
        "onset_times": [round(float(t), 3) for t in onset_times]
    }

def get_energy(y, sr, hop_length=512):
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return [
        {"time": round(float(t), 3), "energy": round(float(e), 4)}
        for t, e in zip(times, rms)
    ]

def get_pitch(y, sr):
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
    )
    times = librosa.times_like(f0, sr=sr)
    return [
        {"time": round(float(t), 3), "hz": round(float(f), 2) if not np.isnan(f) else None}
        for t, f in zip(times, f0)
    ]

def get_spectral_bands(y, sr, hop_length=512):
    stft = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr)
    times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr, hop_length=hop_length)

    bass_mask = (freqs >= 20) & (freqs < 250)
    mid_mask = (freqs >= 250) & (freqs < 4000)
    treble_mask = (freqs >= 4000) & (freqs < 20000)

    bass = stft[bass_mask, :].mean(axis=0)
    mid = stft[mid_mask, :].mean(axis=0)
    treble = stft[treble_mask, :].mean(axis=0)

    return [
        {
            "time": round(float(t), 3),
            "bass": round(float(b), 4),
            "mid": round(float(m), 4),
            "treble": round(float(tr), 4)
        }
        for t, b, m, tr in zip(times, bass, mid, treble)
    ]

def get_silence(y, sr, top_db=30):
    intervals = librosa.effects.split(y, top_db=top_db)
    non_silent = [
        {"start": round(float(s / sr), 3), "end": round(float(e / sr), 3)}
        for s, e in intervals
    ]
    silences = []
    for i in range(len(non_silent) - 1):
        gap_start = non_silent[i]["end"]
        gap_end = non_silent[i + 1]["start"]
        if gap_end - gap_start > 0.05:
            silences.append({"start": gap_start, "end": gap_end})
    return silences

def get_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key_index = int(np.argmax(chroma_mean))
    return keys[key_index]

def get_structure(y, sr):
    S = np.abs(librosa.stft(y))
    bounds = librosa.segment.agglomerative(S, k=6)
    times = librosa.frames_to_time(bounds, sr=sr)
    return [round(float(t), 3) for t in times]

def main(audio_path, output_path, whisper_model="base"):
    y, sr = librosa.load(audio_path, sr=None)

    data = {
        "transcript": get_transcript(audio_path, whisper_model),
        "rhythm": get_rhythm(y, sr),
        "energy": get_energy(y, sr),
        "pitch": get_pitch(y, sr),
        "spectral_bands": get_spectral_bands(y, sr),
        "silence": get_silence(y, sr),
        "key": get_key(y, sr),
        "structure_boundaries": get_structure(y, sr)
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved: {output_path}")

if __name__ == "__main__":
    audio_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "audio_data.json"
    whisper_model = sys.argv[3] if len(sys.argv) > 3 else "base"
    main(audio_path, output_path, whisper_model)
