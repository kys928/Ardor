"""Process-wide HTTP compatibility shim for Ardor control scripts.

GitHub-hosted runners currently receive Cloudflare Error 1010 from
rest.runpod.io when Python urllib sends its default User-Agent. Keep urllib
semantics unchanged, but give Request objects an explicit application
User-Agent unless callers already provided one.
"""
from __future__ import annotations

import urllib.request

_ORIGINAL_REQUEST = urllib.request.Request
_USER_AGENT = "Ardor-RunPod-Control/1.0 (+https://github.com/kys928/Ardor)"


class ArdorRequest(_ORIGINAL_REQUEST):
    def __init__(self, url, data=None, headers=None, origin_req_host=None, unverifiable=False, method=None):
        merged = dict(headers or {})
        if not any(str(key).lower() == "user-agent" for key in merged):
            merged["User-Agent"] = _USER_AGENT
        super().__init__(
            url,
            data=data,
            headers=merged,
            origin_req_host=origin_req_host,
            unverifiable=unverifiable,
            method=method,
        )


urllib.request.Request = ArdorRequest
