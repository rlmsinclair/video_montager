import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import aiofiles
from dotenv import load_dotenv
from google import genai
from openai import AsyncOpenAI
import hashlib
import time
import random

# Load environment variables
load_dotenv()
openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
gemini_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

# Global settings
CHUNK_DURATION = 120  # Duration in seconds per chunk
MAX_CONCURRENT_CHUNKS = 4  # Number of concurrent transcription tasks
MAX_RETRIES = 3  # Number of retries for failed chunks
CACHE_DIR = Path('cache')
CACHE_DIR.mkdir(exist_ok=True)


class ProgressBar:
    def __init__(self, total: int, description: str = ""):
        self.total = total
        self.current = 0
        self.description = description
        self._print_progress()

    def update(self, increment: int = 1, additional_info: str = ""):
        self.current += increment
        self._print_progress(additional_info)

    def _print_progress(self, additional_info: str = ""):
        progress = int(50 * self.current / self.total)
        percentage = (self.current / self.total) * 100
        print(
            f"\r{self.description}[{'=' * progress}{'-' * (50 - progress)}] {self.current}/{self.total} ({percentage:.1f}%){additional_info}",
            end="")
        if self.current >= self.total:
            print()  # New line when complete


class CacheManager:
    @staticmethod
    def get_cache_path(data: str, prefix: str = '', chunk_info: dict = None) -> Path:
        if chunk_info:
            data = f"{data}_{chunk_info.get('start', '')}_{chunk_info.get('duration', '')}"
        hash_str = hashlib.md5(data.encode()).hexdigest()
        return CACHE_DIR / f"{prefix}_{hash_str}.json"

    @staticmethod
    def cleanup_old_cache():
        try:
            current_time = time.time()
            for cache_file in CACHE_DIR.glob('*.json'):
                if current_time - os.path.getmtime(cache_file) > 86400:  # 24 hours
                    try:
                        os.remove(cache_file)
                    except Exception:
                        pass
        except Exception:
            pass

    @staticmethod
    async def read_cache(cache_path: Path) -> Optional[Any]:
        if not cache_path.exists():
            return None
        try:
            async with aiofiles.open(cache_path, 'r') as f:
                content = await f.read()
                if len(content.strip()) < 3:
                    os.remove(cache_path)
                    return None
                return json.loads(content)
        except Exception:
            try:
                os.remove(cache_path)
            except Exception:
                pass
            return None

    @staticmethod
    async def write_cache(cache_path: Path, data: Any):
        try:
            async with aiofiles.open(cache_path, 'w') as f:
                await f.write(json.dumps(data))
        except Exception:
            pass


class FFmpegHelper:
    @staticmethod
    async def run_command(cmd: List[str]) -> Tuple[bool, str]:
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            return process.returncode == 0, stderr.decode()
        except Exception as e:
            return False, str(e)

    @staticmethod
    async def get_duration(file_path: Path) -> Optional[float]:
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(file_path)
            ]
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode == 0 and process.stdout.strip():
                return float(process.stdout.strip())
        except Exception:
            pass
        return None


class FileManager:
    @staticmethod
    async def cleanup_files(paths: List[Path]):
        for path in paths:
            try:
                if path.is_file():
                    os.remove(path)
                elif path.is_dir():
                    for file in path.glob('*'):
                        try:
                            os.remove(file)
                        except Exception:
                            pass
                    try:
                        path.rmdir()
                    except Exception:
                        pass
            except Exception:
                pass


async def split_audio_into_chunks(audio_path: Path, chunk_duration: int = 60) -> List[Path]:
    print("\nSplitting audio into chunks...")
    chunks_dir = Path('temp_audio_chunks')
    chunks_dir.mkdir(exist_ok=True)

    # Check if chunks already exist and are valid
    existing_chunks = sorted(chunks_dir.glob('chunk_*.mp3'))
    if existing_chunks:
        print(f"Found {len(existing_chunks)} existing chunks")
        return existing_chunks

    print("No existing chunks found, creating new ones...")
    try:
        # Get audio duration
        duration = await FFmpegHelper.get_duration(audio_path)
        if not duration:
            print("Could not determine audio duration")
            return []

        print(f"Audio duration: {duration} seconds")
        chunk_paths = []

        # Create chunks in parallel with a semaphore to limit concurrent processes
        semaphore = asyncio.Semaphore(4)  # Limit to 4 concurrent FFmpeg processes

        async def create_chunk(start_time: int) -> Optional[Path]:
            async with semaphore:
                chunk_path = chunks_dir / f'chunk_{start_time}.mp3'
                print(f"\nCreating chunk at {start_time}s -> {chunk_path}")

                cmd = [
                    'ffmpeg', '-i', str(audio_path),
                    '-ss', str(start_time),
                    '-t', str(chunk_duration),
                    '-acodec', 'libmp3lame',
                    '-ac', '1', '-ar', '16000',
                    '-y', str(chunk_path)
                ]

                success, error_msg = await FFmpegHelper.run_command(cmd)
                if success and chunk_path.exists():
                    print(f"Successfully created chunk at {start_time}s")
                    return chunk_path
                else:
                    print(f"Error creating chunk at {start_time}s: {error_msg}")
                    return None

        # Create tasks for all chunks
        tasks = []
        for start_time in range(0, int(duration), chunk_duration):
            tasks.append(create_chunk(start_time))

        # Wait for all chunks to be created
        print(f"Creating {len(tasks)} chunks...")
        results = await asyncio.gather(*tasks)

        # Filter out failed chunks
        chunk_paths = [path for path in results if path is not None]
        print(f"Successfully created {len(chunk_paths)} chunks")

        return sorted(chunk_paths)
    except Exception as e:
        print(f"Error splitting audio into chunks: {e}")
        return []


async def extract_audio_parallel(video_path: Path) -> Optional[Path]:
    print("\nExtracting audio...")
    try:
        duration = await FFmpegHelper.get_duration(video_path)
        if not duration:
            return None

        audio_path = video_path.with_suffix('.mp3')
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vn', '-acodec', 'libmp3lame',
            '-ac', '1', '-ar', '16000',
            '-q:a', '5', '-threads', '4',
            '-y', str(audio_path)
        ]

        success, error_msg = await FFmpegHelper.run_command(cmd)
        if not success:
            print(f"Error extracting audio: {error_msg}")
            return None

        return audio_path if audio_path.exists() else None
    except Exception as e:
        print(f"Error during audio extraction: {e}")
        return None


async def transcribe_chunk(chunk_path: Path, offset: float = 0, chunk_duration: int = 60) -> List[Dict]:
    if not chunk_path.exists() or os.path.getsize(chunk_path) > 25 * 1024 * 1024:
        print(f"\nSkipping chunk {chunk_path} - File too large or doesn't exist")
        return []

    # Check chunk-level cache
    cache_key = f"chunk_{chunk_path.stem}_{os.path.getsize(chunk_path)}_{os.path.getmtime(chunk_path)}"
    cache_path = CacheManager.get_cache_path(cache_key, 'chunk')
    cached_data = await CacheManager.read_cache(cache_path)
    if cached_data:
        print(f"\nUsing cached transcription for chunk {chunk_path}")
        return cached_data

    for attempt in range(MAX_RETRIES):
        try:
            print(f"\nTranscribing chunk: {chunk_path}")
            print(f"Chunk size: {os.path.getsize(chunk_path) / 1024:.2f}KB")

            with open(chunk_path, "rb") as audio_file:
                response = await openai_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    response_format="verbose_json",
                    timestamp_granularities=["word"]
                )

            print(f"\nTranscription successful for chunk {chunk_path}")

            words = []
            if isinstance(response, dict):
                words = (response.get('words', []) or
                         response.get('segments', []) or
                         response.get('word_segments', []))
            elif hasattr(response, 'words'):
                words = response.words
            elif hasattr(response, 'segments'):
                words = response.segments

            result = []
            for word in words:
                try:
                    if isinstance(word, dict):
                        start = float(word.get('start', word.get('start_time', 0)))
                        end = float(word.get('end', word.get('end_time', 0)))
                        text = word.get('word', word.get('text', ''))
                    else:
                        start = float(getattr(word, 'start', getattr(word, 'start_time', 0)))
                        end = float(getattr(word, 'end', getattr(word, 'end_time', 0)))
                        text = getattr(word, 'word', getattr(word, 'text', ''))

                    result.append({
                        'text': text,
                        'start': offset + start,
                        'end': offset + end
                    })
                except Exception:
                    continue

            # Cache successful transcription
            await CacheManager.write_cache(cache_path, result)
            return result
        except Exception as e:
            print(f"\nAttempt {attempt + 1} failed for chunk {chunk_path}: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                retry_delay = 5 * (attempt + 1)
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                print(f"All retries failed for chunk {chunk_path}")
    return []


async def transcribe_audio(audio_path: Path, chunk_duration: int = 60) -> List[Dict]:
    if not audio_path.exists():
        return []

    # Just use the existing audio chunks directly
    chunks = sorted(Path('temp_audio_chunks').glob('chunk_*.mp3'))
    if not chunks:
        print("\nNo audio chunks found!")
        return []

    print(f"\nFound {len(chunks)} existing audio chunks, using directly...")

    all_words = []
    progress = ProgressBar(len(chunks), "Processing chunks: ")

    # Process chunks in parallel
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)

    async def process_chunk(chunk_path: Path) -> List[Dict]:
        async with semaphore:
            try:
                # Get start time from filename (e.g., chunk_60.mp3 -> 60)
                start_time = float(chunk_path.stem.split('_')[1])

                # Check chunk-level cache
                cache_key = f"chunk_{chunk_path.stem}_{os.path.getsize(chunk_path)}_{os.path.getmtime(chunk_path)}"
                cache_path = CacheManager.get_cache_path(cache_key, 'chunk')
                cached_data = await CacheManager.read_cache(cache_path)

                if cached_data:
                    progress.update(1, " (cached)")
                    return cached_data

                # Transcribe if not cached
                with open(chunk_path, "rb") as audio_file:
                    response = await openai_client.audio.transcriptions.create(
                        file=audio_file,
                        model="whisper-1",
                        response_format="verbose_json",
                        timestamp_granularities=["word"]
                    )

                words = []
                if isinstance(response, dict):
                    words = (response.get('words', []) or
                             response.get('segments', []) or
                             response.get('word_segments', []))
                elif hasattr(response, 'words'):
                    words = response.words
                elif hasattr(response, 'segments'):
                    words = response.segments

                result = []
                for word in words:
                    try:
                        if isinstance(word, dict):
                            start = float(word.get('start', word.get('start_time', 0)))
                            end = float(word.get('end', word.get('end_time', 0)))
                            text = word.get('word', word.get('text', ''))
                        else:
                            start = float(getattr(word, 'start', getattr(word, 'start_time', 0)))
                            end = float(getattr(word, 'end', getattr(word, 'end_time', 0)))
                            text = getattr(word, 'word', getattr(word, 'text', ''))

                        result.append({
                            'text': text,
                            'start': start_time + start,
                            'end': start_time + end
                        })
                    except Exception:
                        continue

                # Cache the result
                await CacheManager.write_cache(cache_path, result)
                progress.update()
                return result

            except Exception as e:
                print(f"\nError processing chunk {chunk_path}: {e}")
                progress.update()
                return []

    # Process chunks in batches
    for i in range(0, len(chunks), MAX_CONCURRENT_CHUNKS):
        batch = chunks[i:i + MAX_CONCURRENT_CHUNKS]
        batch_results = await asyncio.gather(*[process_chunk(chunk) for chunk in batch])
        for words in batch_results:
            all_words.extend(words)

    all_words.sort(key=lambda x: x['start'])

    return all_words


def group_words_into_segments(words: List[Dict], segment_duration: int = 120, overlap: int = 20) -> List[Dict]:
    segments = []
    total_duration = words[-1]['end']

    for segment_start in range(0, int(total_duration), segment_duration - overlap):
        segment_words = []
        segment_end = segment_start + segment_duration
        padding = 5

        for word in words:
            if (word['start'] >= segment_start - padding and
                    word['end'] <= segment_end + padding):
                segment_words.append(word)

        if segment_words:
            segments.append({
                'start_time': segment_start,
                'end_time': segment_end,
                'words': segment_words,
                'text': ' '.join(w['text'] for w in segment_words)
            })

    return segments


def merge_overlapping_clips(clips: List[Dict], max_gap: float = 3.0, min_duration: float = 2.5) -> List[Dict]:
    if not clips:
        return []

    sorted_clips = sorted(clips, key=lambda x: x['start_time'])
    merged = []
    current = sorted_clips[0]

    for next_clip in sorted_clips[1:]:
        if next_clip['start_time'] - current['end_time'] <= max_gap:
            current = {
                'start_time': current['start_time'],
                'end_time': next_clip['end_time'],
                'text': current['text'] + ' ' + next_clip['text'],
                'reason': current['reason'] + ' & ' + next_clip['reason'],
                'humor_score': max(current.get('humor_score', 0), next_clip.get('humor_score', 0))
            }
        else:
            if current['end_time'] - current['start_time'] >= min_duration:
                merged.append(current)
            current = next_clip

    if current['end_time'] - current['start_time'] >= min_duration:
        merged.append(current)

    return merged


async def analyze_humor_segments(segments: List[Dict], batch_size: int = 5) -> List[Dict]:
    progress = ProgressBar(len(segments), "Analyzing segments: ")
    semaphore = asyncio.Semaphore(2)  # Limit concurrent API calls to avoid rate limits

    # Group segments into batches
    batches = []
    for i in range(0, len(segments), batch_size):
        batches.append(segments[i:i + batch_size])

    async def analyze_batch(batch: List[Dict]) -> List[Dict]:
        try:
            # Check cache for all segments in batch
            cached_results = []
            segments_to_analyze = []

            for segment in batch:
                cache_path = CacheManager.get_cache_path(
                    segment['text'],
                    'humor',
                    {'start': segment['start_time'], 'end': segment['end_time']}
                )
                print(f"\nChecking humor cache at: {cache_path}")

                cached_data = await CacheManager.read_cache(cache_path)
                if cached_data:
                    print(f"Found cached humor analysis with score: {cached_data.get('humor_score', 0)}")
                    cached_results.append({**segment, **cached_data})
                    progress.update()
                else:
                    segments_to_analyze.append((segment, cache_path))

            if not segments_to_analyze:
                return cached_results

            print(f"\nAnalyzing {len(segments_to_analyze)} segments in batch...")

            async with semaphore:
                try:
                    segments_text = "\n---\n".join(
                        [f"Segment {i + 1}: {s[0]['text']}" for i, s in enumerate(segments_to_analyze)])

                    prompt = f"""You are an advanced narrative analysis system trained to identify compelling story elements and meaningful moments. Analyze these segments for their narrative value:

{segments_text}

Return your analysis as a JSON object with this exact format:
{{
    "segments": [
        {{
            "segment_number": <number of the segment>,
            "score": <number 0-100 indicating narrative importance>,
            "story_moments": [
                {{
                    "text": "<exact word-for-word quote of the important moment>",
                    "reason": "<brief explanation of this moment's narrative significance>"
                }}
            ]
        }}
    ]
}}

Consider ALL narrative elements:
1. Story Development:
- Key plot points
- Character development
- Important revelations
- Setting establishment
- Conflict introduction/resolution

2. Character Elements:
- Personal growth moments
- Relationship dynamics
- Emotional expressions
- Decision points
- Background reveals

3. Thematic Elements:
- Core messages
- Recurring themes
- Symbolic moments
- Value statements
- Life lessons

4. Narrative Techniques:
- Foreshadowing
- Callbacks to earlier events
- World-building details
- Perspective shifts
- Tone-setting moments

5. Emotional Impact:
- Powerful statements
- Genuine reactions
- Meaningful interactions
- Moments of realization
- Personal reflections

Rate based on how each segment contributes to the overall narrative. Consider both explicit story progression and subtle character/thematic development.

Only include moments with clear narrative significance. If a segment doesn't advance the story, return an empty story_moments array.
Ensure the "text" field matches words exactly as they appear in the original text."""

                    # Add retry logic with exponential backoff for rate limits
                    max_retries = 5
                    base_delay = 1

                    for retry in range(max_retries):
                        try:
                            response = await asyncio.to_thread(
                                gemini_client.models.generate_content,
                                model='gemini-2.0-flash-exp',
                                contents=prompt
                            )
                            break
                        except Exception as e:
                            if retry == max_retries - 1:
                                raise e
                            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                                delay = base_delay * (2 ** retry) + random.uniform(0, 1)
                                print(f"\nRate limited, retrying in {delay:.1f} seconds...")
                                await asyncio.sleep(delay)
                                continue
                            raise e

                    # Extract JSON from response
                    text = response.text.strip()
                    start_idx = text.find('{')
                    end_idx = text.rfind('}') + 1

                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = text[start_idx:end_idx]
                    else:
                        print(f"\nNo valid JSON found in response: {text}")
                        json_str = '{"score": 0, "funny_moments": []}'

                    analysis = json.loads(json_str)
                    results = []

                    for segment_analysis in analysis.get('segments', []):
                        segment_num = segment_analysis['segment_number'] - 1
                        if segment_num < 0 or segment_num >= len(segments_to_analyze):
                            continue

                        segment, cache_path = segments_to_analyze[segment_num]
                        funny_clips = []

                        for moment in segment_analysis.get('funny_moments', []):
                            words = moment['text'].split()
                            for i in range(len(segment['words']) - len(words) + 1):
                                segment_words = segment['words'][i:i + len(words)]
                                if ' '.join(w['text'] for w in segment_words).lower() == ' '.join(words).lower():
                                    funny_clips.append({
                                        'start_time': segment_words[0]['start'],
                                        'end_time': segment_words[-1]['end'],
                                        'text': moment['text'],
                                        'reason': moment['reason']
                                    })
                                    break

                        result = {
                            'humor_score': segment_analysis['score'],
                            'funny_clips': funny_clips
                        }

                        print(f"Writing humor analysis to cache: {result}")
                        await CacheManager.write_cache(cache_path, result)
                        results.append({**segment, **result})
                        progress.update()

                    return results + cached_results
                except Exception as e:
                    print(f"\nError analyzing segment: {e}")
                    if hasattr(e, 'response'):
                        print(f"API Response: {e.response}")
                    progress.update()
                    return {**segment, 'humor_score': 0, 'funny_clips': []}
        except Exception as e:
            print(f"\nError in analyze_segment: {e}")
            progress.update()
            return {**segment, 'humor_score': 0, 'funny_clips': []}

    # Process batches in parallel with semaphore controlling concurrency
    tasks = [analyze_batch(batch) for batch in batches]
    batch_results = await asyncio.gather(*tasks)

    # Flatten results
    results = []
    for batch_result in batch_results:
        results.extend(batch_result)

    return results


async def create_montage_parallel(video_path: Path, segments: List[Dict], output_path: Path) -> bool:
    temp_dir = Path('temp_clips')
    temp_dir.mkdir(exist_ok=True)
    progress = ProgressBar(len(segments), "Creating clips: ")

    async def create_clip(segment: Dict, idx: int) -> Optional[Path]:
        try:
            clip_path = temp_dir / f"clip_{idx:03d}.mp4"
            duration = segment['end_time'] - segment['start_time']
            print(
                f"\nCreating clip {idx} from {segment['start_time']:.2f}s to {segment['end_time']:.2f}s (duration: {duration:.2f}s)")
            print(f"Clip text: {segment.get('text', '')}")

            # First detect silence
            silence_detect_cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-ss', str(segment['start_time']),
                '-t', str(duration),
                '-af', 'silencedetect=n=-50dB:d=1',  # Detect silence above 1 second
                '-f', 'null',
                '-'
            ]
            success, stderr = await FFmpegHelper.run_command(silence_detect_cmd)
            
            # Parse silence detection output
            silence_ranges = []
            for line in stderr.split('\n'):
                if 'silence_start' in line:
                    try:
                        start = float(line.split('silence_start: ')[1])
                        silence_ranges.append({'start': start})
                    except Exception:
                        continue
                elif 'silence_end' in line and silence_ranges:
                    try:
                        end = float(line.split('silence_end: ')[1].split(' ')[0])
                        silence_ranges[-1]['end'] = end
                    except Exception:
                        continue

            # Filter out silences less than 1 second
            long_silences = [s for s in silence_ranges if 'end' in s and (s['end'] - s['start']) >= 1]

            if not long_silences:
                # No long silences, use normal clip extraction
                cmd = [
                    'ffmpeg',
                    '-ss', str(segment['start_time']),
                    '-t', str(duration),
                    '-i', str(video_path),
                    '-c:v', 'libx264', '-preset', 'veryfast',
                    '-c:a', 'aac',
                    str(clip_path),
                    '-y'
                ]
            else:
                # Create complex filter to remove silences
                filter_parts = []
                last_end = 0
                temp_files = []
                
                for i, silence in enumerate(long_silences):
                    if silence['start'] > last_end:
                        # Extract segment before silence
                        temp_file = temp_dir / f"temp_{idx}_{i}.mp4"
                        temp_files.append(temp_file)
                        extract_cmd = [
                            'ffmpeg',
                            '-ss', str(segment['start_time'] + last_end),
                            '-t', str(silence['start'] - last_end),
                            '-i', str(video_path),
                            '-c:v', 'libx264', '-preset', 'veryfast',
                            '-c:a', 'aac',
                            str(temp_file),
                            '-y'
                        ]
                        await FFmpegHelper.run_command(extract_cmd)
                    last_end = silence['end']
                
                # Extract final segment after last silence if needed
                if last_end < duration:
                    temp_file = temp_dir / f"temp_{idx}_final.mp4"
                    temp_files.append(temp_file)
                    extract_cmd = [
                        'ffmpeg',
                        '-ss', str(segment['start_time'] + last_end),
                        '-t', str(duration - last_end),
                        '-i', str(video_path),
                        '-c:v', 'libx264', '-preset', 'veryfast',
                        '-c:a', 'aac',
                        str(temp_file),
                        '-y'
                    ]
                    await FFmpegHelper.run_command(extract_cmd)
                
                # Create concat file for the segments
                concat_file = temp_dir / f"concat_{idx}.txt"
                async with aiofiles.open(concat_file, 'w') as f:
                    for temp_file in temp_files:
                        await f.write(f"file '{temp_file.absolute()}'\n")
                
                # Concatenate all segments
                cmd = [
                    'ffmpeg',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', str(concat_file),
                    '-c', 'copy',
                    str(clip_path),
                    '-y'
                ]
            success, error = await FFmpegHelper.run_command(cmd)

            if not success:
                print(f"Error creating clip {idx}: {error}")
                return None

            if not clip_path.exists():
                print(f"Clip file {clip_path} was not created")
                return None

            clip_size = os.path.getsize(clip_path)
            if clip_size == 0:
                print(f"Clip file {clip_path} is empty")
                os.remove(clip_path)
                return None

            print(f"Successfully created clip {idx} ({clip_size / 1024 / 1024:.2f}MB)")
            progress.update()
            return clip_path

        except Exception as e:
            print(f"Exception creating clip {idx}: {str(e)}")
            progress.update()
            return None

    # Process clips sequentially to better handle errors
    clip_paths = []
    for i, segment in enumerate(segments):
        clip_path = await create_clip(segment, i)
        if clip_path:
            clip_paths.append(clip_path)

    if not clip_paths:
        print("\nNo valid clips were created!")
        return False

    print(f"\nSuccessfully created {len(clip_paths)} clips")

    try:
        print("\nMerging clips into final montage...")
        concat_file = temp_dir / 'concat.txt'

        # Write concat file with absolute paths
        async with aiofiles.open(concat_file, 'w') as f:
            for clip_path in clip_paths:
                await f.write(f"file '{clip_path.absolute()}'\n")

        print(f"Created concat file at {concat_file}")
        print("Concat file contents:")
        async with aiofiles.open(concat_file, 'r') as f:
            print(await f.read())

        cmd = [
            'ffmpeg', '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',
            str(output_path),
            '-y'
        ]
        success, error = await FFmpegHelper.run_command(cmd)

        if not success:
            print(f"\nError merging clips: {error}")
        else:
            print("\nMontage creation complete!")

        return success
    finally:
        # Only cleanup after concat is complete
        await FileManager.cleanup_files([temp_dir])


async def main():
    video_path = Path('source.mp4')
    print(f"\nChecking for source video at {video_path}...")
    if not video_path.exists():
        print("Error: source.mp4 not found!")
        return

    output_path = Path('funny_montage.mp4')
    print(f"Checking for existing montage at {output_path}...")
    if output_path.exists():
        print("Funny montage already exists! Skipping all processing.")
        return

    try:
        print("\nStarting funny montage creation...")

        # Check for existing audio file
        audio_path = video_path.with_suffix('.mp3')
        print(f"Checking for existing audio at {audio_path}...")
        if not audio_path.exists():
            print("No existing audio found. Extracting audio...")
            audio_path = await extract_audio_parallel(video_path)
            if not audio_path:
                print("Error extracting audio!")
                return
        else:
            print("Found existing audio file, skipping extraction...")

        # Split audio into chunks
        print("\nSplitting audio into chunks...")
        chunks = await split_audio_into_chunks(audio_path)
        if not chunks:
            print("Error splitting audio into chunks!")
            return

        # Check cache for transcription
        cache_key = f"transcription_{audio_path.stem}_{os.path.getsize(audio_path)}_{os.path.getmtime(audio_path)}"
        cache_path = CacheManager.get_cache_path(cache_key, 'transcription')
        print(f"Checking for cached transcription at {cache_path}...")
        cached_transcription = await CacheManager.read_cache(cache_path)

        if cached_transcription:
            print("\nFound cached transcription, skipping transcription process...")
            words = cached_transcription
        else:
            print("\nNo cached transcription found, starting full transcription process...")
            words = await transcribe_audio(audio_path)
            if not words:
                print("Error transcribing audio!")
                return

        print("\nProcessing segments from transcription...")
        segments = group_words_into_segments(words)

        # Check if all segments are already analyzed and cached
        all_cached = True
        cached_segments = []

        for segment in segments:
            cache_path = CacheManager.get_cache_path(
                segment['text'],
                'humor',
                {'start': segment['start_time'], 'end': segment['end_time']}
            )
            cached_data = await CacheManager.read_cache(cache_path)
            if cached_data:
                cached_segments.append({**segment, **cached_data})
            else:
                all_cached = False
                break

        if all_cached:
            print("\nUsing cached humor analysis for all segments...")
            analyzed_segments = cached_segments
        else:
            print("\nAnalyzing segments for humor...")
            analyzed_segments = await analyze_humor_segments(segments)
        print(f"Found {len(analyzed_segments)} analyzed segments")

        story_clips = []
        for segment in analyzed_segments:
            narrative_score = segment.get('score', 0)
            clips = segment.get('story_moments', [])
            print(f"\nSegment narrative score: {narrative_score}")
            print(f"Found {len(clips)} story moments in segment")
            
            # Weight different types of narrative moments
            for clip in clips:
                adjusted_score = narrative_score
                reason = clip.get('reason', '').lower()
                
                # Boost for key story elements
                if any(word in reason for word in ['plot', 'revelation', 'conflict', 'resolution']):
                    adjusted_score += 15
                
                # Boost for character development
                if any(word in reason for word in ['character', 'growth', 'relationship', 'emotional']):
                    adjusted_score += 10
                
                # Boost for thematic elements
                if any(word in reason for word in ['theme', 'message', 'symbolic', 'lesson']):
                    adjusted_score += 10
                
                clip['adjusted_score'] = min(100, adjusted_score)
                
                # Include significant story moments
                if adjusted_score >= 50:  # Lower threshold to catch more narrative elements
                    story_clips.append(clip)

        print(f"\nTotal story moments found: {len(story_clips)}")
        if story_clips:
            # Sort by timestamp to maintain chronological order
            story_clips.sort(key=lambda x: x['start_time'])
            merged_clips = merge_overlapping_clips(story_clips)
            final_clips = merged_clips[:30]

            success = await create_montage_parallel(
                video_path,
                final_clips,
                output_path
            )

            if success:
                print("Successfully created funny_montage.mp4!")
            else:
                print("Error creating montage!")
        else:
            print("No funny clips found!")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Only cleanup temporary files, not the audio file since we want to reuse it
        await FileManager.cleanup_files([
            Path('temp_chunks'),
            Path('temp_clips')
        ])


if __name__ == "__main__":
    asyncio.run(main())
