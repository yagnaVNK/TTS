#!/usr/bin/env python3
"""
Script to resample audio files to 8kHz for faster TTS processing
Run this script to convert your speaker reference files to 8kHz
"""

import os
import librosa
import soundfile as sf
from pathlib import Path

def resample_audio_file(input_path, output_path, target_sr=8000):
    """
    Resample an audio file to the target sample rate
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save resampled audio file
        target_sr: Target sample rate (default: 8000Hz)
    """
    try:
        # Load audio file
        audio, orig_sr = librosa.load(input_path, sr=None)
        print(f"Loaded {input_path}: {orig_sr}Hz -> {target_sr}Hz")
        
        # Resample if needed
        if orig_sr != target_sr:
            audio_resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        else:
            audio_resampled = audio
            print(f"  No resampling needed, already {target_sr}Hz")
        
        # Save resampled audio
        sf.write(output_path, audio_resampled, target_sr)
        print(f"  Saved to {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def main():
    """Main function to resample all speaker files"""
    
    # Define the audio files to resample
    audio_files = [
        {
            'input': 'default_audio/english_audio2.wav',
            'output': 'default_audio/english_audio2_8k.wav'
        },
        {
            'input': 'default_audio/japanese.wav',
            'output': 'default_audio/japanese_8k.wav'
        },
        {
            'input': 'default_audio/spanish_audio1.wav',
            'output': 'default_audio/spanish_audio1_8k.wav'
        }
    ]
    
    target_sample_rate = 8000
    
    print(f"Resampling audio files to {target_sample_rate}Hz...")
    print("=" * 50)
    
    # Create output directory if it doesn't exist
    os.makedirs('default_audio', exist_ok=True)
    
    success_count = 0
    total_count = len(audio_files)
    
    for file_info in audio_files:
        input_path = file_info['input']
        output_path = file_info['output']
        
        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found, skipping...")
            continue
            
        if resample_audio_file(input_path, output_path, target_sample_rate):
            success_count += 1
        
        print()
    
    print("=" * 50)
    print(f"Completed: {success_count}/{total_count} files processed successfully")
    
    if success_count == total_count:
        print("✅ All files resampled successfully!")
        print("\nYou can now use the updated backend code with 8kHz sampling rate.")
    else:
        print("⚠️  Some files failed to process. Check the errors above.")

if __name__ == "__main__":
    main()