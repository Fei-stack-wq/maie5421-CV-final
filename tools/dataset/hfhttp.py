"""Minimal HTTP range-backed seekable file for reading remote Parquet metadata/columns."""
import io
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (dataset-builder)"}


class HttpRangeFile:
    def __init__(self, url, timeout=120, cache_limit=256 * 1024 * 1024):
        self.url = url
        self.timeout = timeout
        self.pos = 0
        self.size = None
        self._cache = []  # list of (start, end, bytes)
        self._cache_bytes = 0
        self._cache_limit = cache_limit
        self.closed = False
        self._head()

    def _open(self, headers=None):
        h = dict(UA)
        if headers:
            h.update(headers)
        req = urllib.request.Request(self.url, headers=h)
        return urllib.request.urlopen(req, timeout=self.timeout)

    def _head(self):
        req = urllib.request.Request(self.url, headers=dict(UA, Range="bytes=0-0"))
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            cr = r.headers.get("Content-Range")
            if cr and "/" in cr:
                self.size = int(cr.split("/")[-1])

    def _fetch(self, start, end):
        # end exclusive
        with self._open({"Range": "bytes=%d-%d" % (start, end - 1)}) as r:
            data = r.read()
        if len(data) != end - start:
            raise IOError("short read %d != %d" % (len(data), end - start))
        return data

    def _cached(self, start, end):
        for (s, e, b) in self._cache:
            if s <= start and end <= e:
                return b[start - s:end - s]
        return None

    def _store(self, start, b):
        self._cache.append((start, start + len(b), b))
        self._cache_bytes += len(b)
        while self._cache_bytes > self._cache_limit and len(self._cache) > 1:
            s, e, old = self._cache.pop(0)
            self._cache_bytes -= len(old)

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size - self.pos
        end = min(self.pos + size, self.size)
        if end <= self.pos:
            return b""
        data = self._cached(self.pos, end)
        if data is None:
            data = self._fetch(self.pos, end)
            self._store(self.pos, data)
        self.pos = end
        return data

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def readable(self):
        return True

    def seekable(self):
        return True

    def writable(self):
        return False

    def close(self):
        self.closed = True

    def flush(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False
