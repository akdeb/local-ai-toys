#!/usr/bin/env python3
"""
Standalone test script for Chatterbox Turbo TTS (MLX implementation).

Usage:
    python test_cb_turbo.py --ref_audio my_voice_2min.wav --text "Hello, this is a test."
    python test_cb_turbo.py --ref_audio my_voice_2min.wav  # Uses default text
"""

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description="Test Chatterbox Turbo TTS")
    parser.add_argument(
        "--ref_audio",
        type=str,
        required=True,
        help="Path to reference audio file (must be >5 seconds)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello! This is a test of the Chatterbox Turbo text to speech system running on Apple Silicon with MLX.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cb_turbo_output2.wav",
        help="Output WAV file path",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1000,
        help="Top-k sampling (default: 1000)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling (default: 0.95)",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help="Repetition penalty (default: 1.2)",
    )
    parser.add_argument(
        "--local_path",
        type=str,
        default=None,
        help="Path to local checkpoint (optional, downloads from HF if not provided)",
    )
    args = parser.parse_args()

    # Validate reference audio exists
    ref_path = Path(args.ref_audio)
    if not ref_path.exists():
        print(f"Error: Reference audio file not found: {args.ref_audio}")
        return 1

    print("=" * 60)
    print("Chatterbox Turbo TTS Test")
    print("=" * 60)
    print(f"Reference audio: {args.ref_audio}")
    print(f"Text: {args.text}")
    print(f"Output: {args.output}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Top-p: {args.top_p}")
    print("=" * 60)

    # Import here to show loading progress
    print("\nLoading model...")
    load_start = time.time()

    from mlx_audio.tts.utils import load_model

    if args.local_path:
        # Load from local checkpoint
        from cb_turbo import ChatterboxTurboTTS
        model = ChatterboxTurboTTS.from_local(args.local_path)
    else:
        # Load from HuggingFace via mlx_audio
        model = load_model("mlx-community/chatterbox-turbo-4bit")

    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    # Generate speech
    print("\nGenerating speech...")
    gen_start = time.time()

    for result in model.generate(
        text=args.text,
        ref_audio=args.ref_audio,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    ):
        # Convert to numpy and save
        audio = np.array(result.audio)
        sf.write(args.output, audio, result.sample_rate)

        gen_time = time.time() - gen_start
        print("\n" + "=" * 60)
        print("Generation Complete!")
        print("=" * 60)
        print(f"Output saved to: {args.output}")
        print(f"Audio duration: {result.audio_duration}")
        print(f"Sample rate: {result.sample_rate} Hz")
        print(f"Samples: {result.samples}")
        print(f"Processing time: {result.processing_time_seconds:.2f}s")
        print(f"Real-time factor: {result.real_time_factor}x")
        print(f"Peak memory: {result.peak_memory_usage:.2f} GB")
        print(f"Token count: {result.token_count}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
