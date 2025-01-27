# Video Story Moment Extractor

A Python tool that automatically identifies and extracts meaningful narrative moments from videos using advanced audio transcription and AI analysis.

## Features

- Parallel audio extraction and processing for improved performance
- Word-level transcription using OpenAI's Whisper API
- Narrative analysis using Google's Gemini API
- Intelligent caching system to save API calls and processing time
- Automatic silence detection and removal
- Smart clip merging with configurable thresholds
- Progress bars for visual feedback

## Prerequisites

- Python 3.8+
- FFmpeg installed and available in system PATH
- OpenAI API key
- Google Gemini API key

### Required Python Packages

```bash
pip install aiofiles python-dotenv google-generativeai openai
```

## Setup

1. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
```

2. Place your source video as `source.mp4` in the project directory

## Usage

Run the script:
```bash
python script.py
```

The script will:
1. Extract audio from the video
2. Split audio into manageable chunks
3. Transcribe audio using Whisper API
4. Analyze transcriptions for narrative significance using Gemini
5. Create a montage of meaningful moments

The final output will be saved as `funny_montage.mp4`.

## How It Works

### Audio Processing
- Extracts audio from video using FFmpeg
- Splits audio into 120-second chunks for parallel processing
- Uses adaptive retry logic for failed chunks

### Transcription
- Utilizes OpenAI's Whisper API for accurate word-level transcription
- Implements parallel processing with concurrency limits
- Caches results to prevent redundant API calls

### Narrative Analysis
- Uses Gemini AI to identify significant story moments
- Considers multiple narrative elements:
  - Story Development (plot points, revelations)
  - Character Elements (growth, relationships)
  - Thematic Elements (messages, symbolism)
  - Narrative Techniques (foreshadowing, callbacks)
  - Emotional Impact (powerful statements, reactions)

### Clip Generation
- Implements intelligent silence detection
- Merges overlapping clips with configurable thresholds
- Removes long silences for better pacing
- Creates final montage using FFmpeg

## Configuration

Key settings can be adjusted at the top of the script:
```python
CHUNK_DURATION = 120  # Duration in seconds per chunk
MAX_CONCURRENT_CHUNKS = 4  # Number of concurrent transcription tasks
MAX_RETRIES = 3  # Number of retries for failed chunks
```

## Caching

The tool implements a sophisticated caching system:
- Caches transcriptions at both chunk and full-file levels
- Caches narrative analysis results
- Automatically cleans up cache files older than 24 hours
- Uses MD5 hashing for cache keys

## Error Handling

- Implements exponential backoff for rate limits
- Retries failed API calls with configurable attempts
- Graceful handling of missing or corrupt files
- Comprehensive error logging

## File Structure

```
project/
├── .env                    # API keys
├── source.mp4             # Input video
├── funny_montage.mp4      # Output video
├── cache/                 # Cache directory
├── temp_audio_chunks/     # Temporary audio chunks
└── temp_clips/           # Temporary video clips
```

## Performance Considerations

- Memory usage scales with chunk size and concurrency
- API costs depend on video length and chunk settings
- Disk space needed for temporary files and cache

## Error Codes and Troubleshooting

Common issues and solutions:
- Missing FFmpeg: Install FFmpeg and ensure it's in system PATH
- API rate limits: Adjust MAX_CONCURRENT_CHUNKS
- Memory issues: Reduce CHUNK_DURATION
- Missing files: Check file permissions and paths

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for Whisper API
- Google for Gemini API
- FFmpeg contributors