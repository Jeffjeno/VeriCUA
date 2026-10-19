from __future__ import annotations
import hashlib
import os
import re
import socket
from functools import lru_cache

def wa_site_env_defaults() -> dict[str, str]:

    host = os.environ.get('WA_EXTERNAL_HOST') or socket.getfqdn()
    offset = int(os.environ.get('WA_INSTANCE', '0')) * int(os.environ.get('WA_PORT_STRIDE', '1000'))
    ro_offset = 0 if os.environ.get('WA_SHARE_READONLY', '1') == '1' else offset
    return {'WA_SHOPPING': f'http://{host}:{7770 + offset}', 'WA_SHOPPING_ADMIN': f'http://{host}:{7780 + offset}/admin', 'WA_REDDIT': f'http://{host}:{9999 + offset}', 'WA_GITLAB': f'http://{host}:{8023 + offset}', 'WA_WIKIPEDIA': f'http://{host}:{8888 + ro_offset}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing', 'WA_MAP': f'http://{host}:{3000 + ro_offset}', 'WA_HOMEPAGE': f'http://{host}:{4399 + offset}', 'WA_FULL_RESET': f'http://{host}:{7565 + offset}'}
ACTION_URL_RE = re.compile('(?P<scheme>https?)://(?P<host>[^/\\s\'\\"()]+):(?P<port>\\d+)(?=[/\\s\'\\"()<>]|$)')

def rewrite_webarena_action_urls(action: str) -> str:

    host = os.environ.get('WA_EXTERNAL_HOST') or socket.getfqdn()
    stride = int(os.environ.get('WA_PORT_STRIDE', '1000'))
    target_urls = wa_site_env_defaults()

    def port_from(url: str) -> int:
        match = re.search(':(\\d+)(?:/|$)', url)
        if match is None:
            raise ValueError(f'WebArena URL has no port: {url}')
        return int(match.group(1))
    target_ports = {7770: port_from(target_urls['WA_SHOPPING']), 7780: port_from(target_urls['WA_SHOPPING_ADMIN']), 9999: port_from(target_urls['WA_REDDIT']), 8023: port_from(target_urls['WA_GITLAB']), 8888: port_from(target_urls['WA_WIKIPEDIA']), 3000: port_from(target_urls['WA_MAP']), 4399: port_from(target_urls['WA_HOMEPAGE'])}

    def source_port_family(port: int) -> int | None:
        if port == 25099:
            return 9999
        if port in {3000, 3001, 8080}:
            return 3000
        if port == 8888 or (port > 8888 and (port - 8888) % stride == 0):
            return 8888
        for base in (7770, 7780, 9999, 8023, 4399):
            if port >= base and (port - base) % stride == 0:
                return base
        return None

    def replace(match: re.Match[str]) -> str:
        family = source_port_family(int(match.group('port')))
        if family is None:
            return match.group(0)
        target_port = target_ports[family]
        return f"{match.group('scheme')}://{host}:{target_port}"
    return ACTION_URL_RE.sub(replace, action)

def stable_shard(value: str, num_shards: int) -> int:
    if num_shards < 1:
        raise ValueError('num_shards must be positive')
    digest = hashlib.sha256(value.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') % num_shards

@lru_cache(maxsize=4)
def build_action_set(kind: str):
    import bgym
    if kind == 'holo3':
        return bgym.HighLevelActionSet(subsets=['chat', 'infeas', 'bid', 'coord', 'nav', 'tab'], multiaction=True, strict=True)
    if kind == 'generic_webarena':
        return bgym.HighLevelActionSet(subsets=['webarena'], multiaction=False, strict=False, retry_with_force=True, demo_mode='off')
    raise ValueError(f'Unknown action set kind: {kind}')
