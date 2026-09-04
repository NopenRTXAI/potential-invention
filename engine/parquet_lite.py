"""Minimal parquet reader for the DailyDialog dataset (pure stdlib + numpy).

Parquet is a nested columnar format; a full reader is huge. This one
supports exactly what our dataset file needs: plain + dictionary encoding,
PLAIN and RLE/BIT-PACKED hybrid byte arrays, snappy block compression
(we decompress with the 'snappy' module if available, else fail loudly).

Used only to load conversation data. Not a general parquet reader.
"""

import io
import json
import struct
import numpy as np


# ---------------------------------------------------------------- thrift

def read_list_header(buf, pos):
    """Compact-protocol list header: (size<<4)|type, or 0xF + varint size."""
    b = buf[pos]; pos += 1
    sz = (b >> 4) & 0x0F
    elem_type = b & 0x0F
    if sz == 15:
        sz, pos = read_varint(buf, pos)
    return sz, elem_type, pos


def read_thrift(buf, pos):
    """Read one thrift compact-protocol struct -> (dict, new_pos).

    Returns nested dict of field_id -> value. Enough for footer metadata.
    """
    out = {}
    last_fid = 0
    while True:
        b = buf[pos]; pos += 1
        if b == 0:  # stop
            return out, pos
        ftype = b & 0x0F
        fid = (b & 0xF0) >> 4
        if fid == 0:
            fid = read_varint(buf, pos)[0]
            pos = read_varint(buf, pos)[1]
        else:
            fid = last_fid + fid
        last_fid = fid
        if ftype == 1:  # bool_true
            out[fid] = True
        elif ftype == 2:  # bool_false
            out[fid] = False
        elif ftype == 3:  # i8
            out[fid] = struct.unpack_from(">b", buf, pos)[0]; pos += 1
        elif ftype == 4:  # i16
            v, pos = read_varint(buf, pos); out[fid] = v
        elif ftype == 5:  # i32
            v, pos = read_varint(buf, pos); out[fid] = v
        elif ftype == 6:  # i64
            v, pos = read_varint(buf, pos); out[fid] = v
        elif ftype == 7:  # double
            out[fid] = struct.unpack_from(">d", buf, pos)[0]; pos += 8
        elif ftype == 8:  # binary
            ln, pos = read_varint(buf, pos)
            out[fid] = bytes(buf[pos:pos + ln]); pos += ln
        elif ftype == 9:  # list
            sz, elem_type, pos = read_list_header(buf, pos)
            vals = []
            for _ in range(sz):
                v, pos = read_thrift_elem(buf, pos, elem_type)
                vals.append(v)
            out[fid] = vals
        elif ftype == 12:  # struct
            sub, pos = read_thrift(buf, pos)
            out[fid] = sub
        else:
            raise ValueError(f"thrift type {ftype} unsupported")
    return out, pos


def read_thrift_elem(buf, pos, ftype):
    if ftype == 8:
        ln, pos = read_varint(buf, pos)
        return bytes(buf[pos:pos + ln]), pos + ln
    if ftype == 5 or ftype == 6:
        v, pos = read_varint(buf, pos); return v, pos
    if ftype == 12:
        return read_thrift(buf, pos)
    if ftype == 9:
        sz, elem_type, pos = read_list_header(buf, pos)
        vals = []
        for _ in range(sz):
            v, pos = read_thrift_elem(buf, pos, elem_type)
            vals.append(v)
        return vals, pos
    if ftype == 3:
        return struct.unpack_from(">b", buf, pos)[0], pos + 1
    raise ValueError(f"thrift elem type {ftype} unsupported")


def read_varint(buf, pos):
    result = 0; shift = 0
    while True:
        b = buf[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


# ---------------------------------------------------------------- snappy

def snappy_decompress(data):
    """Pure-python snappy raw-format decompressor (no framing)."""
    try:
        import snappy as _snappy
        return _snappy.decompress(data)
    except ImportError:
        pass
    out = bytearray()
    pos = 0
    # uncompressed length varint
    length = 0; shift = 0
    while True:
        b = data[pos]; pos += 1
        length |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    while pos < len(data):
        b = data[pos]; pos += 1
        tag = b & 0x03
        if tag == 0:  # literal
            ln = (b >> 2) + 1
            if ln > 60:
                extra = ln - 60
                ln = int.from_bytes(data[pos:pos + extra], "little") + 1
                pos += extra
            out += data[pos:pos + ln]; pos += ln
        elif tag == 1:  # copy with 1-byte offset
            ln = ((b >> 2) & 0x07) + 4
            off = ((b >> 5) << 8) | data[pos]; pos += 1
            for _ in range(ln):
                out.append(out[-off])
        elif tag == 2:  # copy 2-byte offset
            ln = (b >> 2) + 1
            off = int.from_bytes(data[pos:pos + 2], "little"); pos += 2
            for _ in range(ln):
                out.append(out[-off])
        elif tag == 3:  # copy 4-byte offset
            ln = (b >> 2) + 1
            off = int.from_bytes(data[pos:pos + 4], "little"); pos += 4
            for _ in range(ln):
                out.append(out[-off])
    if len(out) != length:
        raise ValueError(f"snappy length mismatch {len(out)} != {length}")
    return bytes(out)


# ---------------------------------------------------------------- parquet

def read_parquet(path):
    """Return list of row-dicts for a small parquet file."""
    raw = open(path, "rb").read()
    if raw[:4] != b"PAR1" or raw[-4:] != b"PAR1":
        raise ValueError("not a parquet file")
    footer_len = struct.unpack_from("<I", raw, len(raw) - 8)[0]
    fstart = len(raw) - 8 - footer_len
    footer, _ = read_thrift(raw, fstart)

    schema_elems = footer[2]      # list of SchemaElement
    row_groups = footer[4]        # list of RowGroup
    num_rows = footer[3]

    # schema: element 0 is root; children follow in order
    cols = []  # (name, type)
    for el in schema_elems[1:]:
        if 4 in el:  # num_children -> root-ish, skip
            continue
        cols.append((el[5].decode() if isinstance(el[5], bytes) else el[5],
                     el.get(1)))  # name, type
    rows = [dict() for _ in range(num_rows)]
    for rg in row_groups:
        for col in rg[1]:  # ColumnChunk
            meta = col[3]          # ColumnMetaData
            name = meta[3][0].decode()   # path_in_schema[0]
            ctype = meta[1]        # type
            data_off = meta[5]      # data_page_offset
            csize = meta[7]        # total_compressed_size
            blob = raw[data_off:data_off + csize]
            values = decode_column(blob, meta, ctype, num_rows)
            for i, v in enumerate(values):
                rows[i][name] = v
    return rows


def decode_column(blob, meta, ctype, num_rows):
    """Decode one column chunk (dictionary + plain/RLE data pages)."""
    pos = 0
    dict_vals = None
    out = []
    while pos < len(blob):
        page_len = struct.unpack_from("<I", blob, pos)[0]
        page = blob[pos + 4: pos + 4 + page_len]
        pos += 4 + page_len
        ptype = page[0] & 0xFF  # first byte of page header (thrift)
        # parse page header thrift
        hdr, hpos = read_thrift(page, 0)
        ptype = hdr[1]  # page type: 0=data,2=dictionary,3=levels
        unc = hdr[2]   # uncompressed_page_size
        comp = hdr[3]  # compressed_page_size
        # compression codec: 0=uncompressed,1=snappy
        codec = meta[8] if 8 in meta else 0
        body = page[hpos:hpos + comp]
        if codec == 1:
            body = snappy_decompress(body)
        if ptype == 2:  # dictionary page
            n = hdr[5]  # num_values
            dict_vals = decode_plain(body, ctype, n)
        elif ptype == 0:  # data page v1
            n = hdr[5]
            if dict_vals is not None:
                # RLE/bit-packed hybrid dictionary indices
                # def levels first? v1: definition levels RLE precede if
                # the column is optional. Our dataset columns are all
                # required (no nulls), so skip.
                bw = body[0]  # bit width byte
                idxs = decode_rle_bp(body[1:], bw, n)
                out.extend(dict_vals[i] for i in idxs)
            else:
                out.extend(decode_plain(body, ctype, n))
    return out


def decode_plain(body, ctype, n):
    if ctype == 6:  # INT64
        return list(struct.unpack_from(f"<{n}q", body, 0))
    if ctype == 7:  # INT96 (timestamp) — not used here
        return [None] * n
    if ctype == 0:  # boolean
        return [bool(body[i // 8] & (1 << (i % 8))) for i in range(n)]
    if ctype == 2:  # INT32
        return list(struct.unpack_from(f"<{n}i", body, 0))
    if ctype == 5:  # FLOAT
        return list(struct.unpack_from(f"<{n}f", body, 0))
    if ctype == 3:  # INT96
        return [None] * n
    raise ValueError(f"plain type {ctype} unsupported")


def decode_rle_bp(body, bit_width, n):
    """RLE / bit-packed hybrid decoder."""
    vals = []
    pos = 0
    total_bits = bit_width * n
    while len(vals) < n:
        header, pos = read_varint(body, pos)
        if header & 1:  # bit-packed run
            groups = header >> 1
            nvals = groups * 8
            nbytes = groups * bit_width
            chunk = body[pos:pos + nbytes]; pos += nbytes
            bits = np.frombuffer(chunk, dtype=np.uint8)
            # unpack bit_width bits little-endian per value
            vals.extend(unpack_bits(bits, bit_width, nvals))
        else:  # RLE run
            run_len = header >> 1
            nbytes = (bit_width + 7) // 8
            v = int.from_bytes(body[pos:pos + nbytes], "little")
            pos += nbytes
            vals.extend([v] * run_len)
    return vals[:n]


def unpack_bits(bits, bw, n):
    """Unpack n values of bw bits each (LSB-first bit order)."""
    out = []
    bitpos = 0
    total = len(bits) * 8
    for _ in range(n):
        v = 0
        for k in range(bw):
            byte_i = (bitpos + k) // 8
            bit_i = (bitpos + k) % 8
            if byte_i < len(bits):
                v |= ((bits[byte_i] >> bit_i) & 1) << k
        out.append(v)
        bitpos += bw
    return out
