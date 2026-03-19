from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

from google import genai
from dotenv import load_dotenv


# For quick iteration, keep this configurable from the environment.
# Example alternatives:
# - veo-3.1-generate-preview: latest quality, preview
# - veo-3.1-fast-generate-preview: latest fast preview
# - veo-3.0-generate-001: stable higher-quality option
# - veo-3.0-fast-generate-001: stable faster option
load_dotenv()
MODEL_NAME = os.getenv("GOOGLE_VIDEO_MODEL", "veo-3.1-fast-generate-preview")

PROMPT = """
초여름 저녁의 서울 골목.
따뜻한 주황빛 가로등 아래에서 한 사람이 투명한 우산을 들고 천천히 걷는다.
카메라는 영화 같은 시네마틱 무드로 뒤에서 따라가다가,
마지막에는 옆모습 클로즈업으로 전환된다.
잔잔한 바람에 옷자락이 흔들리고, 비에 젖은 바닥에는 도시 불빛이 반사된다.
현실적인 영상, 자연스러운 인물 움직임, 부드러운 카메라 무빙, 16:9, 8초 분량.
""".strip()


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")

    output_dir = Path("generated_videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=api_key)

    print(f"Using model: {MODEL_NAME}")
    print("Submitting video generation request...")

    # Keep this request close to Google's official example.
    # The model already defaults to a single 16:9 video, and unsupported
    # config fields can cause 400 INVALID_ARGUMENT errors.
    operation = client.models.generate_videos(
        model=MODEL_NAME,
        prompt=PROMPT,
    )

    while not operation.done:
        print("Waiting for video generation to complete...")
        time.sleep(10)
        operation = client.operations.get(operation)

    if not operation.response or not operation.response.generated_videos:
        raise RuntimeError("No video was returned by the API.")

    generated_video = operation.response.generated_videos[0]
    client.files.download(file=generated_video.video)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"veo-{timestamp}.mp4"
    generated_video.video.save(str(output_path))

    print(f"Saved video to: {output_path.resolve()}")
    print("Open the generated_videos directory to inspect the downloaded mp4 file.")
    if getattr(generated_video.video, "uri", None):
        print(f"Remote video URI: {generated_video.video.uri}")


if __name__ == "__main__":
    main()
