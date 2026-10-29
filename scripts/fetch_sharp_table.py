"""Extract Sharp's 32 KB timestep table locally or with bounded HTTP ranges.

This is an explicit setup command; the nodes never download files.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.request


SOURCE = ('https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/'
          'e5eb578a89295337b8ff433a035929ce0279e0b6/diffusion_models/'
          'minimax_h3_ref2va_pruned_int8_convrot.safetensors')
SHA256 = 'c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7'


def remote_range(start, length):
    end = start + length - 1
    # Separate cache entries for each range; some CDN caches otherwise reuse one.
    request = urllib.request.Request(SOURCE + f'?range={start}-{end}', headers={
        'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
    with urllib.request.urlopen(request, timeout=60) as response:
        content_range = response.headers.get('Content-Range', '')
        if response.status != 206 or not content_range.startswith(f'bytes {start}-{end}/'):
            raise RuntimeError('Server ignored the byte range; refusing a full checkpoint download')
        data = response.read(length + 1)
    if len(data) != length:
        raise RuntimeError('Incomplete or oversized byte-range response')
    return data


def extract(read_range):
    size = struct.unpack('<Q', read_range(0, 8))[0]
    if not 2 <= size <= 8 * 1024**2:
        raise ValueError('Invalid or unexpectedly large safetensors header')
    header = json.loads(read_range(8, size))
    tensor = header['adaln_t_table']
    start, end = tensor['data_offsets']
    if tensor['dtype'] != 'F32' or tensor['shape'] != [1025, 8] or end - start != 32800 or start < 0:
        raise ValueError('Unexpected Sharp timestep tensor')
    data = read_range(8 + size + start, end - start)
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError('Sharp timestep table checksum mismatch')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='Destination adaln_t_table.bin')
    parser.add_argument('--source-model', type=Path, help='Existing native INT8 Ref2VA safetensors; otherwise fetch only the table')
    args = parser.parse_args()
    if (args.output.is_file() and args.output.stat().st_size == 32800
            and hashlib.sha256(args.output.read_bytes()).hexdigest() == SHA256):
        print(f'Already verified: {args.output}')
        return
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite an unexpected file: {args.output}')
    if args.source_model:
        with args.source_model.open('rb') as source:
            def read_range(start, length):
                source.seek(start)
                data = source.read(length)
                if len(data) != length:
                    raise ValueError('Truncated source checkpoint')
                return data
            data = extract(read_range)
    else:
        data = extract(remote_range)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as output:
        output.write(data)
    print(f'Verified and wrote {len(data):,} bytes: {args.output}')


if __name__ == '__main__':
    main()
