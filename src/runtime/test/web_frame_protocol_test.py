#!/usr/bin/env python3
import mmap
import tempfile
import time

import web


def write_frame(mm, *, generation, cache_seq, capture_seq, detector_seq, cache_timestamp_ns):
    jpeg = b"\xff\xd8\xff\xd9"
    header = web.FRAME_HEADER.pack(
        web.FRAME_MAGIC,
        web.FRAME_VERSION,
        web.FRAME_HEADER.size,
        generation,
        len(jpeg),
        0,
        cache_seq,
        capture_seq,
        detector_seq,
        123456789,
        cache_timestamp_ns,
    )
    mm.seek(0)
    mm.write(header)
    mm.write(jpeg)
    mm.flush()


def main():
    with tempfile.TemporaryFile() as backing:
        backing.truncate(web.shared_size)
        with mmap.mmap(backing.fileno(), web.shared_size) as mm:
            web.mapfile = mm
            web.use_shared_memory = True
            now_ns = time.time_ns()
            write_frame(
                mm,
                generation=2,
                cache_seq=7,
                capture_seq=1247,
                detector_seq=1024,
                cache_timestamp_ns=now_ns,
            )
            frame = web.read_jpeg_from_shared_memory()
            assert frame is not None
            jpeg, metadata = frame
            assert jpeg == b"\xff\xd8\xff\xd9"
            assert metadata["cache_seq"] == 7
            assert metadata["capture_seq"] == 1247
            assert metadata["detector_seq"] == 1024

            write_frame(
                mm,
                generation=3,
                cache_seq=8,
                capture_seq=1248,
                detector_seq=1025,
                cache_timestamp_ns=now_ns,
            )
            assert web.read_jpeg_from_shared_memory() is None

    web.mapfile = None
    web.use_shared_memory = False
    web.latest_seq = 9
    web.latest_frame_status = {
        "capture_seq": 1247,
        "detector_seq": 1024,
        "cache_seq": 7,
        "capture_timestamp_ns": 123456789,
        "cache_timestamp_ns": time.time_ns() - 1_000_000_000,
    }
    client = web.app.test_client()
    status = client.get("/frame_status").get_json()
    assert status["web_seq"] == 9
    assert status["stale"] is True
    assert status["cache_age_ms"] >= 500
    print("WEB_FRAME_PROTOCOL_TEST PASS")


if __name__ == "__main__":
    main()
