import subprocess
import re


def find_internal_camera():
    result = subprocess.run(
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    output = result.stderr  # ffmpeg listet auf stderr

    # Suche nach Einträgen, die FaceTime enthalten
    matches = re.findall(r"\[(\d+)\] .*FaceTime.*", output)
    print("matches: ", matches)
    if matches:
        return int(matches[0])
    else:
        raise RuntimeError("❌ FaceTime HD-Kamera nicht gefunden")
