"""只读解析本地固件 FAT16/FAT32，提取关键点模型，不修改镜像。"""
import struct
import hashlib
from pathlib import Path


def extract(source, destination):
    with source.open('rb') as disk:
        def read(offset, size):
            disk.seek(offset)
            return disk.read(size)
        mbr = read(0, 512)
        for part in range(4):
            entry = mbr[446 + part * 16:462 + part * 16]
            if entry[4] not in (11, 12):
                continue
            base = struct.unpack_from('<I', entry, 8)[0] * 512
            boot = read(base, 512)
            sector = struct.unpack_from('<H', boot, 11)[0]
            cluster_bytes = sector * boot[13]
            reserved = struct.unpack_from('<H', boot, 14)[0]
            fat16 = struct.unpack_from('<H', boot, 22)[0]
            fat_size = fat16 or struct.unpack_from('<I', boot, 36)[0]
            root_bytes = struct.unpack_from('<H', boot, 17)[0] * 32 if fat16 else 0
            fat = read(base + reserved * sector, fat_size * sector)
            data = base + (reserved + boot[16] * fat_size) * sector
            root_data = read(data, root_bytes) if fat16 else None
            data += ((root_bytes + sector - 1) // sector) * sector
            root = 0 if fat16 else struct.unpack_from('<I', boot, 44)[0]

            def chain(cluster):
                if cluster == 0 and fat16:
                    yield root_data
                    return
                seen = set()
                step = 2 if fat16 else 4
                while 2 <= cluster < (0xfff8 if fat16 else 0x0ffffff8):
                    if cluster in seen or cluster * step + step > len(fat):
                        raise ValueError('Invalid FAT chain')
                    seen.add(cluster)
                    yield read(data + (cluster - 2) * cluster_bytes, cluster_bytes)
                    cluster = struct.unpack_from('<H' if fat16 else '<I', fat, cluster * step)[0] & 0x0fffffff

            def walk(cluster, prefix='', depth=0):
                if depth > 20:
                    return
                longname = []
                for block in chain(cluster):
                    for pos in range(0, len(block), 32):
                        e = block[pos:pos + 32]
                        if e[0] == 0:
                            return
                        if e[0] == 229:
                            longname = []
                            continue
                        if e[11] == 15:
                            raw = e[1:11] + e[14:26] + e[28:32]
                            longname.insert(0, raw.decode('utf-16le').split('\x00')[0].replace('\uffff', ''))
                            continue
                        name = ''.join(longname) if longname else e[:8].decode('ascii', 'replace').strip() + ('.' + e[8:11].decode('ascii', 'replace').strip() if e[8:11].strip() else '')
                        longname = []
                        if e[11] & 8 or name in ('.', '..'):
                            continue
                        start = (struct.unpack_from('<H', e, 20)[0] << 16) | struct.unpack_from('<H', e, 26)[0]
                        if name.lower() == 'face_landmark.kmodel':
                            size = struct.unpack_from('<I', e, 28)[0]
                            payload = b''.join(chain(start))[:size]
                            if len(payload) != size or not size:
                                raise ValueError('Incomplete model')
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            destination.write_bytes(payload)
                            print('Extracted', prefix + name, size, 'bytes', hashlib.sha256(payload).hexdigest())
                            return True
                        if e[11] & 16 and walk(start, prefix + name + '/', depth + 1):
                            return True
                return False
            if walk(root):
                return
    raise FileNotFoundError('face_landmark.kmodel not found in FAT partitions')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    source = next(p for p in root.rglob('*.img') if p.is_file())
    extract(source, root / 'MyVisionHub/dist/models/face_landmark.kmodel')
