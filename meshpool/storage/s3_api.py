"""S3-compatible protocol adapter for Mesh Store (task p5-s3).

Route prefix: /s3  (mounted by controlplane/app.py).

Operations
----------
GET  /s3/                               ListBuckets
PUT  /s3/{bucket}                       CreateBucket
DELETE /s3/{bucket}                     DeleteBucket
GET  /s3/{bucket}?list-type=2           ListObjectsV2
GET  /s3/{bucket}                       GetBucketLocation (minimal)
PUT  /s3/{bucket}/{key}                 PutObject
GET  /s3/{bucket}/{key}                 GetObject  (Range supported -> 206)
HEAD /s3/{bucket}/{key}                 HeadObject
DELETE /s3/{bucket}/{key}               DeleteObject
POST /s3/{bucket}/{key}?uploads         CreateMultipartUpload
PUT  /s3/{bucket}/{key}?partNumber&uploadId UploadPart
POST /s3/{bucket}/{key}?uploadId        CompleteMultipartUpload
DELETE /s3/{bucket}/{key}?uploadId      AbortMultipartUpload

Error responses use S3 XML: <Error><Code>...</Code><Message>...</Message></Error>
ETag: MD5(content) for single-part; "MD5(part_md5s)-N" for multipart.

Auth
----
In auth_mode=open (the default LAN / sim mode) the CP middleware already granted an
anonymous full-access principal, so every request is accepted without any S3
credential check.  In auth_mode=enforced the middleware validated a MeshPool
principal (session token or capability token) or a SigV4 principal resolved by
_get_request_principal in app.py.  Either way, by the time a handler runs,
request.state.principal is set.

SigV4 credential store
----------------------
CP.s3_credentials maps access_key_id -> (secret_key, owner_subject).
Credentials are created via POST /api/v1/s3/credentials and are in-memory only
(lost on restart); the gate creates them fresh each run.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from fastapi import APIRouter, Request
from fastapi.responses import Response

from .engine import StorageError

# S3 XML namespace
_S3_NS = "http://s3.amazonaws.com/doc/2006-03-01/"

# Module-level state reset by tests via _reset_for_tests()
_s3_obj_meta: dict[tuple[str, str], dict[str, Any]] = {}   # (bucket, key) -> metadata
_mp_uploads: dict[str, dict[str, Any]] = {}                 # uploadId -> upload state

# CP accessor injected by app.py via init()
_get_cp: Optional[Callable] = None


def init(get_cp_fn: Callable) -> None:
    """Wire the CP accessor.  Called once from _register_routes in app.py."""
    global _get_cp
    _get_cp = get_cp_fn


def _reset_for_tests() -> None:
    """Clear module-level state between unit tests."""
    _s3_obj_meta.clear()
    _mp_uploads.clear()


def _cp():
    if _get_cp is None:
        raise RuntimeError("s3_api.init() was not called")
    cp = _get_cp()
    if cp is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="control plane not ready")
    return cp


def _owner(request: Request) -> str:
    """Resolve MeshPool owner from request.state.principal."""
    p = getattr(request.state, "principal", None)
    if p is None or getattr(p, "subject", "anonymous") == "anonymous":
        return "default"
    return p.subject


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def _s3_err(code: str, message: str, status: int) -> Response:
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Error>'
        f'<Code>{code}</Code>'
        f'<Message>{message}</Message>'
        '</Error>'
    )
    return Response(content=xml, status_code=status, media_type="application/xml")


def _xml_resp(root: ET.Element, status: int = 200) -> Response:
    ET.register_namespace("", _S3_NS)
    body = b'<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(
        root, encoding="unicode").encode("utf-8")
    return Response(content=body, status_code=status, media_type="application/xml")


def _ts(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _http_date(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%a, %d %b %Y %H:%M:%S GMT")


# ---------------------------------------------------------------------------
# Spend-cap check (inline to avoid circular import at module load time)
# ---------------------------------------------------------------------------

def _check_credits(cp, owner: str) -> Optional[Response]:
    """Return an S3 AccessDenied response if the owner is spend-capped, else None."""
    try:
        if cp.spend_caps.exceeded(owner):
            st = cp.spend_caps.status(owner)
            return _s3_err("AccessDenied",
                            f"Spend cap reached for {owner}: "
                            f"{st['spent']} of {st['cap']} units used.",
                            403)
        if not cp.accounts.can_spend(owner):
            return _s3_err("AccessDenied",
                            f"Insufficient compute credits for {owner}.",
                            403)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Range header parsing
# ---------------------------------------------------------------------------

def _parse_range(hdr: str, total: int) -> Optional[tuple[int, int]]:
    """Parse 'bytes=N-M' into (start, end) inclusive.  Returns None on error."""
    if not hdr or not hdr.startswith("bytes="):
        return None
    spec = hdr[6:]
    try:
        if spec.startswith("-"):
            n = int(spec)
            return max(0, total + n), total - 1
        if spec.endswith("-"):
            start = int(spec[:-1])
            if start >= total:
                return None
            return start, total - 1
        lo, hi = spec.split("-", 1)
        start, end = int(lo), int(hi)
        if start > end or start >= total:
            return None
        return start, min(end, total - 1)
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# SigV4 verification (used when auth_mode=enforced and AWS credentials presented)
# ---------------------------------------------------------------------------

_SIGV4_RE = re.compile(
    r'AWS4-HMAC-SHA256\s+'
    r'Credential=(?P<access_key>[^/]+)/(?P<date>\d{8})/(?P<region>[^/]+)/(?P<service>[^/]+)/aws4_request'
    r',\s*SignedHeaders=(?P<signed_hdrs>[^,]+)'
    r',\s*Signature=(?P<signature>[0-9a-f]+)',
    re.IGNORECASE,
)


def _hmac256(key: bytes, msg: bytes) -> bytes:
    return hmac.new(key, msg, hashlib.sha256).digest()


def _derive_key(secret: str, date: str, region: str, service: str) -> bytes:
    k = _hmac256(("AWS4" + secret).encode(), date.encode())
    k = _hmac256(k, region.encode())
    k = _hmac256(k, service.encode())
    return _hmac256(k, b"aws4_request")


def verify_sigv4_request(request: Request, cp) -> Optional[Any]:
    """Parse AWS SigV4 Authorization header, look up the access key in
    cp.s3_credentials, and return a MeshPool Principal if the signature is valid.
    Returns None if the header is absent, unparseable, or the signature fails.
    """
    auth = request.headers.get("authorization", "")
    m = _SIGV4_RE.match(auth)
    if not m:
        return None

    access_key = m.group("access_key")
    creds = getattr(cp, "s3_credentials", {}).get(access_key)
    if creds is None:
        return None
    secret_key, owner = creds

    date_str = m.group("date")
    region = m.group("region")
    service = m.group("service")
    signed_hdrs_str = m.group("signed_hdrs")
    given_sig = m.group("signature")

    # Canonical request
    method = request.method.upper()
    path = request.url.path
    # Canonical URI: percent-encode path segments but not /
    canonical_uri = "/" + "/".join(
        _uri_encode(seg) for seg in path.lstrip("/").split("/"))

    # Canonical query string: sorted by key then value, URL-encoded
    params = sorted(request.query_params.multi_items())
    canonical_qs = "&".join(
        f"{_uri_encode(k)}={_uri_encode(v)}" for k, v in params)

    # Canonical headers: only the signed ones, lowercased name, stripped value
    signed_names = [h.lower() for h in signed_hdrs_str.split(";")]
    hdrs: dict[str, str] = {}
    for name in signed_names:
        val = request.headers.get(name, "")
        hdrs[name] = " ".join(val.split())  # collapse whitespace
    canonical_headers = "".join(f"{n}:{hdrs.get(n, '')}\n" for n in sorted(signed_names))
    signed_headers_canonical = ";".join(sorted(signed_names))

    # Payload hash: taken from x-amz-content-sha256 header (avoids buffering body)
    payload_hash = request.headers.get("x-amz-content-sha256", "UNSIGNED-PAYLOAD")

    canonical_req = "\n".join([
        method, canonical_uri, canonical_qs,
        canonical_headers, signed_headers_canonical, payload_hash,
    ])

    # String to sign
    amz_date = request.headers.get("x-amz-date", date_str + "T000000Z")
    credential_scope = f"{date_str}/{region}/{service}/aws4_request"
    cr_hash = hashlib.sha256(canonical_req.encode()).hexdigest()
    string_to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, credential_scope, cr_hash])

    # Signature
    signing_key = _derive_key(secret_key, date_str, region, service)
    expected = _hmac256(signing_key, string_to_sign.encode()).hex()

    if not hmac.compare_digest(expected, given_sig):
        return None

    # Return a MeshPool principal for this owner
    from ..common.auth import mint_session_principal
    return mint_session_principal(owner)


def _uri_encode(s: str) -> str:
    """Percent-encode a string per AWS SigV4 rules (unreserved chars pass through)."""
    safe = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
    return "".join(c if c in safe else f"%{ord(c):02X}" for c in s)


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

router = APIRouter()


# ─── GET /  ListBuckets ────────────────────────────────────────────────────
# Handle both GET /s3 and GET /s3/ — botocore._urljoin strips the trailing
# slash when requestUri='/' and endpoint_url doesn't end with '/'.

@router.get("")
@router.get("/")
async def list_buckets(request: Request) -> Response:
    cp = _cp()
    actor = _owner(request)
    try:
        buckets = cp.storage.list_all_buckets(actor=actor)
    except Exception:
        buckets = []
    root = ET.Element("ListAllMyBucketsResult", xmlns=_S3_NS)
    owner_el = ET.SubElement(root, "Owner")
    ET.SubElement(owner_el, "ID").text = actor
    ET.SubElement(owner_el, "DisplayName").text = actor
    buckets_el = ET.SubElement(root, "Buckets")
    for b in sorted(buckets, key=lambda x: x["bucket"]):
        bel = ET.SubElement(buckets_el, "Bucket")
        ET.SubElement(bel, "Name").text = b["bucket"]
        ET.SubElement(bel, "CreationDate").text = _ts(b.get("created_ms", 0))
    return _xml_resp(root)


# ─── PUT /{bucket}  CreateBucket ──────────────────────────────────────────

@router.put("/{bucket}")
async def create_bucket(bucket: str, request: Request) -> Response:
    cp = _cp()
    actor = _owner(request)
    try:
        cp.storage.create_bucket(bucket, actor)
    except StorageError as exc:
        return _s3_err("InvalidBucketName", str(exc), 400)
    return Response(status_code=200)


# ─── DELETE /{bucket}  DeleteBucket ──────────────────────────────────────

@router.delete("/{bucket}")
async def delete_bucket(bucket: str, request: Request) -> Response:
    cp = _cp()
    actor = _owner(request)
    try:
        ok = cp.storage.delete_bucket(bucket=bucket, actor=actor)
    except PermissionError as exc:
        return _s3_err("AccessDenied", str(exc), 403)
    except StorageError as exc:
        msg = str(exc)
        if "not empty" in msg.lower():
            return _s3_err("BucketNotEmpty",
                           "The bucket you tried to delete is not empty.", 409)
        return _s3_err("NoSuchBucket", "The specified bucket does not exist.", 404)
    if not ok:
        return _s3_err("NoSuchBucket", "The specified bucket does not exist.", 404)
    return Response(status_code=204)


# ─── GET /{bucket}  ListObjectsV2 (or GetBucketLocation) ─────────────────

@router.get("/{bucket}")
async def get_bucket(bucket: str, request: Request) -> Response:
    cp = _cp()
    actor = _owner(request)
    qs = dict(request.query_params)

    if "list-type" in qs or "prefix" in qs or "delimiter" in qs or "max-keys" in qs:
        return await _list_objects_v2(cp, actor, bucket, qs)

    # GetBucketLocation or other bucket-level GETs — return minimal response
    root = ET.Element("LocationConstraint", xmlns=_S3_NS)
    root.text = ""
    return _xml_resp(root)


async def _list_objects_v2(cp, actor: str, bucket: str, qs: dict) -> Response:
    try:
        objects = cp.storage.list_bucket_objects(actor=actor, bucket=bucket)
    except StorageError:
        return _s3_err("NoSuchBucket", "The specified bucket does not exist.", 404)
    except PermissionError as exc:
        return _s3_err("AccessDenied", str(exc), 403)

    prefix = qs.get("prefix", "")
    delimiter = qs.get("delimiter", "")
    try:
        max_keys = min(1000, max(1, int(qs.get("max-keys", "1000"))))
    except (ValueError, TypeError):
        max_keys = 1000
    cont_token = qs.get("continuation-token", "")
    start_after = qs.get("start-after", "")

    objects.sort(key=lambda o: o["key"])
    filtered = [o for o in objects if o["key"].startswith(prefix)]

    start_from = cont_token or start_after
    if start_from:
        filtered = [o for o in filtered if o["key"] > start_from]

    common_prefixes: set[str] = set()
    contents: list[dict] = []
    if delimiter:
        for o in filtered:
            rest = o["key"][len(prefix):]
            idx = rest.find(delimiter)
            if idx >= 0:
                common_prefixes.add(prefix + rest[: idx + len(delimiter)])
            else:
                contents.append(o)
    else:
        contents = filtered

    is_truncated = len(contents) > max_keys
    contents = contents[:max_keys]
    next_token = contents[-1]["key"] if is_truncated else ""

    root = ET.Element("ListBucketResult", xmlns=_S3_NS)
    ET.SubElement(root, "Name").text = bucket
    ET.SubElement(root, "Prefix").text = prefix
    ET.SubElement(root, "KeyCount").text = str(len(contents))
    ET.SubElement(root, "MaxKeys").text = str(max_keys)
    ET.SubElement(root, "IsTruncated").text = str(is_truncated).lower()
    if is_truncated:
        ET.SubElement(root, "NextContinuationToken").text = next_token
    if delimiter:
        ET.SubElement(root, "Delimiter").text = delimiter
    if cont_token:
        ET.SubElement(root, "ContinuationToken").text = cont_token

    for o in contents:
        meta = _s3_obj_meta.get((bucket, o["key"]), {})
        etag = meta.get("etag") or f'"{hashlib.md5(b"").hexdigest()}"'
        cel = ET.SubElement(root, "Contents")
        ET.SubElement(cel, "Key").text = o["key"]
        ET.SubElement(cel, "Size").text = str(meta.get("size", o.get("size", 0)))
        ET.SubElement(cel, "ETag").text = etag
        ET.SubElement(cel, "LastModified").text = _ts(
            meta.get("last_modified", o.get("created_ms", 0)))
        ET.SubElement(cel, "StorageClass").text = "STANDARD"

    for cp_str in sorted(common_prefixes):
        cp_el = ET.SubElement(root, "CommonPrefixes")
        ET.SubElement(cp_el, "Prefix").text = cp_str

    return _xml_resp(root)


# ─── PUT /{bucket}/{key}  PutObject or UploadPart ─────────────────────────

@router.put("/{bucket}/{key:path}")
async def put_object(bucket: str, key: str, request: Request) -> Response:
    cp = _cp()
    actor = _owner(request)
    qs = dict(request.query_params)

    if "uploadId" in qs and "partNumber" in qs:
        return await _upload_part(bucket, key, qs, request)

    data = await request.body()
    if not data:
        # S3 allows empty PutObject (zero-byte file)
        data = b""

    content_type = request.headers.get("content-type", "application/octet-stream")
    etag = f'"{hashlib.md5(data).hexdigest()}"'

    cap_err = _check_credits(cp, actor)
    if cap_err:
        return cap_err

    try:
        await cp.storage.put_object_version(
            data, actor=actor, bucket=bucket, key=key,
            name=f"{bucket}/{key}")
    except PermissionError as exc:
        return _s3_err("AccessDenied", str(exc), 403)
    except StorageError as exc:
        return _s3_err("ServiceUnavailable", str(exc), 503)

    _s3_obj_meta[(bucket, key)] = {
        "etag": etag,
        "content_type": content_type,
        "last_modified": int(time.time() * 1000),
        "size": len(data),
    }
    return Response(status_code=200, headers={"ETag": etag})


async def _upload_part(bucket: str, key: str, qs: dict,
                       request: Request) -> Response:
    upload_id = qs.get("uploadId", "")
    if upload_id not in _mp_uploads:
        return _s3_err("NoSuchUpload", "The specified upload does not exist.", 404)
    try:
        part_num = int(qs.get("partNumber", "0"))
    except (ValueError, TypeError):
        return _s3_err("InvalidArgument", "PartNumber must be an integer.", 400)
    if not (1 <= part_num <= 10000):
        return _s3_err("InvalidArgument",
                       "PartNumber must be between 1 and 10000.", 400)
    data = await request.body()
    etag = f'"{hashlib.md5(data).hexdigest()}"'
    _mp_uploads[upload_id]["parts"][part_num] = {"data": data, "etag": etag}
    return Response(status_code=200, headers={"ETag": etag})


# ─── GET /{bucket}/{key}  GetObject ──────────────────────────────────────

@router.get("/{bucket}/{key:path}")
async def get_object(bucket: str, key: str, request: Request) -> Response:
    cp = _cp()
    actor = _owner(request)

    try:
        data = await cp.storage.get_object_version(
            actor=actor, bucket=bucket, key=key)
    except PermissionError as exc:
        return _s3_err("AccessDenied", str(exc), 403)
    except StorageError as exc:
        return _s3_err("NoSuchBucket", str(exc), 404)

    if data is None:
        return _s3_err("NoSuchKey", "The specified key does not exist.", 404)

    meta = _s3_obj_meta.get((bucket, key), {})
    etag = meta.get("etag") or f'"{hashlib.md5(data).hexdigest()}"'
    ctype = meta.get("content_type", "application/octet-stream")
    lm_ms = meta.get("last_modified", int(time.time() * 1000))

    range_hdr = request.headers.get("range", "")
    if range_hdr:
        r = _parse_range(range_hdr, len(data))
        if r is None:
            return _s3_err("InvalidRange",
                           "The requested range is not satisfiable.", 416)
        start, end = r
        body = data[start: end + 1]
        return Response(
            content=body, status_code=206, media_type=ctype,
            headers={
                "ETag": etag,
                "Content-Type": ctype,
                "Last-Modified": _http_date(lm_ms),
                "Content-Range": f"bytes {start}-{end}/{len(data)}",
                "Content-Length": str(len(body)),
            },
        )

    return Response(
        content=data, status_code=200, media_type=ctype,
        headers={
            "ETag": etag,
            "Content-Type": ctype,
            "Last-Modified": _http_date(lm_ms),
            "Content-Length": str(len(data)),
        },
    )


# ─── HEAD /{bucket}/{key}  HeadObject ─────────────────────────────────────

@router.head("/{bucket}/{key:path}")
async def head_object(bucket: str, key: str, request: Request) -> Response:
    cp = _cp()
    actor = _owner(request)

    try:
        objects = cp.storage.list_bucket_objects(actor=actor, bucket=bucket)
    except PermissionError as exc:
        return Response(status_code=403)
    except StorageError:
        return Response(status_code=404)

    obj = next((o for o in objects if o["key"] == key), None)
    if obj is None:
        return Response(status_code=404)

    meta = _s3_obj_meta.get((bucket, key), {})
    etag = meta.get("etag", "")
    ctype = meta.get("content_type", "application/octet-stream")
    size = meta.get("size", obj.get("size", 0))
    lm_ms = meta.get("last_modified", obj.get("created_ms", 0))

    return Response(
        status_code=200,
        headers={
            "ETag": etag,
            "Content-Type": ctype,
            "Content-Length": str(size),
            "Last-Modified": _http_date(lm_ms),
        },
    )


# ─── DELETE /{bucket}/{key}  DeleteObject or AbortMultipartUpload ─────────

@router.delete("/{bucket}/{key:path}")
async def delete_object(bucket: str, key: str, request: Request) -> Response:
    cp = _cp()
    actor = _owner(request)
    qs = dict(request.query_params)

    if "uploadId" in qs:
        _mp_uploads.pop(qs["uploadId"], None)
        return Response(status_code=204)

    try:
        await cp.storage.delete_object_version(actor=actor, bucket=bucket, key=key)
    except PermissionError as exc:
        return _s3_err("AccessDenied", str(exc), 403)
    except StorageError as exc:
        return _s3_err("NoSuchBucket", str(exc), 404)

    _s3_obj_meta.pop((bucket, key), None)
    return Response(status_code=204)


# ─── POST /{bucket}/{key}  CreateMultipartUpload or CompleteMultipartUpload

@router.post("/{bucket}/{key:path}")
async def post_object(bucket: str, key: str, request: Request) -> Response:
    cp = _cp()
    actor = _owner(request)
    qs = dict(request.query_params)

    if "uploads" in qs:
        return _initiate_multipart(bucket, key, request, actor)

    if "uploadId" in qs:
        return await _complete_multipart(cp, actor, bucket, key, qs, request)

    return _s3_err("NotImplemented", "Unknown POST operation.", 501)


def _initiate_multipart(bucket: str, key: str,
                        request: Request, actor: str) -> Response:
    upload_id = secrets.token_hex(16)
    _mp_uploads[upload_id] = {
        "bucket": bucket,
        "key": key,
        "actor": actor,
        "content_type": request.headers.get("content-type", "application/octet-stream"),
        "parts": {},
    }
    root = ET.Element("InitiateMultipartUploadResult", xmlns=_S3_NS)
    ET.SubElement(root, "Bucket").text = bucket
    ET.SubElement(root, "Key").text = key
    ET.SubElement(root, "UploadId").text = upload_id
    return _xml_resp(root)


async def _complete_multipart(cp, actor: str, bucket: str, key: str,
                               qs: dict, request: Request) -> Response:
    upload_id = qs.get("uploadId", "")
    if upload_id not in _mp_uploads:
        return _s3_err("NoSuchUpload", "The specified upload does not exist.", 404)
    upload = _mp_uploads[upload_id]
    if upload["bucket"] != bucket or upload["key"] != key:
        return _s3_err("InvalidRequest",
                       "Upload ID does not match this object.", 400)

    body = await request.body()
    try:
        xml_root = ET.fromstring(body.decode("utf-8"))
        ns_prefix = ""
        if xml_root.tag.startswith("{"):
            ns_prefix = xml_root.tag.split("}")[0] + "}"
        parts_spec: list[tuple[int, str]] = []
        for pel in xml_root.findall(f"{ns_prefix}Part"):
            pn = pel.find(f"{ns_prefix}PartNumber")
            et = pel.find(f"{ns_prefix}ETag")
            if pn is not None and et is not None:
                parts_spec.append((int(pn.text), et.text.strip('"')))
        parts_spec.sort(key=lambda x: x[0])
    except Exception:
        return _s3_err("MalformedXML",
                       "The XML you provided was not well-formed.", 400)

    if not parts_spec:
        return _s3_err("MalformedXML", "You must specify at least one part.", 400)

    parts_store = upload["parts"]
    assembled = b""
    for pn, _ in parts_spec:
        if pn not in parts_store:
            return _s3_err("InvalidPart", f"Part {pn} was not uploaded.", 400)
        assembled += parts_store[pn]["data"]

    # Multipart ETag: MD5 of concatenated binary MD5s of each part, then "-N"
    raw_md5s = b""
    for pn, _ in parts_spec:
        raw_md5s += bytes.fromhex(parts_store[pn]["etag"].strip('"'))
    final_etag = f'"{hashlib.md5(raw_md5s).hexdigest()}-{len(parts_spec)}"'
    ctype = upload.get("content_type", "application/octet-stream")

    cap_err = _check_credits(cp, actor)
    if cap_err:
        return cap_err

    try:
        await cp.storage.put_object_version(
            assembled, actor=actor, bucket=bucket, key=key,
            name=f"{bucket}/{key}")
    except PermissionError as exc:
        return _s3_err("AccessDenied", str(exc), 403)
    except StorageError as exc:
        return _s3_err("ServiceUnavailable", str(exc), 503)

    _s3_obj_meta[(bucket, key)] = {
        "etag": final_etag,
        "content_type": ctype,
        "last_modified": int(time.time() * 1000),
        "size": len(assembled),
    }
    del _mp_uploads[upload_id]

    result = ET.Element("CompleteMultipartUploadResult", xmlns=_S3_NS)
    ET.SubElement(result, "Location").text = f"/{bucket}/{key}"
    ET.SubElement(result, "Bucket").text = bucket
    ET.SubElement(result, "Key").text = key
    ET.SubElement(result, "ETag").text = final_etag
    return _xml_resp(result)
