"""Model providers. All take (system, parts) where parts is a list of str | bytes (JPEG) and return text.

  claude-cli  Anthropic *subscription* through the local `claude -p` (no API key; ~1 s for haiku)
  anthropic   Anthropic API (env ANTHROPIC_API_KEY)
  openai      any OpenAI-compatible /chat/completions endpoint: OpenAI, OpenRouter, Ollama, LM Studio, vLLM, llama.cpp
              (key from env OPENAI_API_KEY or the env var named in api_key_env; optional for local servers)

API keys are only ever read from the environment, never stored.
"""
import base64, hashlib, json, os, subprocess, tempfile, threading, time, urllib.error, urllib.request


class ProviderError(Exception):
    pass


class _CacheBreak:
    """Sentinel in `parts`: everything before it is the stable prefix (system + task + demo) and should be cached."""
    def __repr__(self): return "CACHE_BREAK"


CACHE_BREAK = _CacheBreak()


def _usage(u, ms=None):
    """Normalize provider usage to {input, cache_read, cache_write, output, ms}; input = uncached remainder."""
    u = u or {}
    out = {"input": u.get("input_tokens", u.get("prompt_tokens")),
           "cache_read": u.get("cache_read_input_tokens", (u.get("prompt_tokens_details") or {}).get("cached_tokens")),
           "cache_write": u.get("cache_creation_input_tokens"), "output": u.get("output_tokens", u.get("completion_tokens"))}
    if ms is not None: out["ms"] = ms
    return {k: v for k, v in out.items() if v is not None}


PRESETS = {
    "claude-haiku":     dict(provider="claude-cli", model="haiku"),
    "claude-sonnet":    dict(provider="claude-cli", model="sonnet"),
    "api-haiku":        dict(provider="anthropic", model="claude-haiku-4-5-20251001"),
    "api-sonnet":       dict(provider="anthropic", model="claude-sonnet-5-5"),
    "openai":           dict(provider="openai", model="gpt-4o", base_url="https://api.openai.com/v1"),
    "gemma-e4b":        dict(provider="openai", model="gemma4:e4b-mlx", base_url="http://localhost:11434/v1", api_key_env="NO_KEY_NEEDED"),
    "gemma-12b":        dict(provider="openai", model="gemma4:12b-mlx", base_url="http://localhost:11434/v1", api_key_env="NO_KEY_NEEDED"),
    "gemma-31b":        dict(provider="openai", model="gemma4:31b-mlx", base_url="http://localhost:11434/v1", api_key_env="NO_KEY_NEEDED"),
    "ollama":           dict(provider="openai", model="gemma4:12b-mlx", base_url="http://localhost:11434/v1", api_key_env="NO_KEY_NEEDED"),
    "lmstudio":         dict(provider="openai", model="local-model", base_url="http://localhost:1234/v1"),
}


def _b64(jpeg):
    return base64.b64encode(jpeg).decode()


def _anthropic_content(parts, cache=True):
    out = []
    for p in parts:
        if p is CACHE_BREAK:
            if cache and out: out[-1] = {**out[-1], "cache_control": {"type": "ephemeral"}}
        elif isinstance(p, str):
            out.append({"type": "text", "text": p})
        else:
            out.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(p)}})
    return out


class ClaudeCLI:
    """Anthropic subscription via `claude -p`.

    The CLI ignores our cache marker and caches only a whole identical prompt, so to reuse the demo/task prefix we do what the
    CLI supports: PRIME a stored session once with the stable prefix, then every tick FORK that session (--resume --fork-session)
    and send only the changing tail. The fork starts from the primed history, so the prefix is a cache read (measured: ~16k tokens
    read, ~12 billed per tick) and nothing accumulates between ticks (each fork is discarded; no memory).
    """
    def __init__(self, model="haiku", timeout=120, effort="low", **_):
        self.model, self.timeout, self.effort, self.name = model, timeout, effort, f"claude-cli:{model}"
        self.usage, self._primed, self._sessions = {}, None, []

    @staticmethod
    def _cwd():
        # Fixed empty cwd: project CLAUDE.md is not loaded and the path is identical every call (a changing path breaks the cache)
        cwd = os.path.join(tempfile.gettempdir(), "evg-claude-cwd"); os.makedirs(cwd, exist_ok=True); return cwd

    def _run(self, system, parts, extra=(), persist=False):
        msg = {"type": "user", "message": {"role": "user", "content": _anthropic_content(parts, cache=False)}}
        cmd = ["claude", "-p", "--model", self.model, "--input-format", "stream-json", "--output-format", "stream-json",
               "--verbose", "--tools", "", "--system-prompt", system, *(["--effort", self.effort] if self.effort else []), *extra]
        if not persist and "--resume" not in extra:
            cmd.append("--no-session-persistence")
        try:
            env = dict(os.environ)
            if self.effort == "low" and "haiku" in self.model:
                env["MAX_THINKING_TOKENS"] = "0"        # Haiku has no effort control; its default extended thinking costs 5-20 s per tick
            p = subprocess.Popen(cmd, cwd=self._cwd(), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except FileNotFoundError:
            raise ProviderError("`claude` CLI not found in PATH (needed for subscription mode)")
        killer = threading.Timer(self.timeout, p.kill); killer.start()
        try:
            p.stdin.write(json.dumps(msg) + "\n"); p.stdin.flush()     # stdin must stay open until the result arrives
            for line in p.stdout:
                try: d = json.loads(line)
                except ValueError: continue
                if d.get("type") == "result":
                    if d.get("is_error"):
                        raise ProviderError(f"claude error: {str(d.get('result'))[:300]}")
                    if d.get("session_id"): self._sessions.append(d["session_id"])
                    self.usage = {**_usage(d.get("usage"), d.get("duration_ms")), "usd_equiv": d.get("total_cost_usd")}
                    return d
            raise ProviderError(f"claude produced no result: {p.stderr.read()[:300]}")
        finally:
            killer.cancel()
            try: p.stdin.close()
            except Exception: pass
            p.terminate()

    def complete(self, system, parts, schema=None):
        cut = next((i for i, p in enumerate(parts) if p is CACHE_BREAK), None)
        if cut is None:
            return self._run(system, parts).get("result") or ""
        prefix, tail = parts[:cut], parts[cut + 1:]
        h = hashlib.sha256(system.encode())
        for p in prefix: h.update(p.encode() if isinstance(p, str) else p)
        key = h.hexdigest()
        try:
            if not self._primed or self._primed[0] != key:
                self._primed = None
                r = self._run(system, prefix + ["Acknowledge with the single word READY."], persist=True)
                self._primed = (key, r["session_id"]); prime_usage = self.usage
            r = self._run(system, tail, ["--resume", self._primed[1], "--fork-session"], persist=True)
            return r.get("result") or ""
        except ProviderError:
            self._primed = None          # stale/missing session: fall back to the full prompt this time
            return self._run(system, [p for p in parts if p is not CACHE_BREAK]).get("result") or ""

    def close(self):
        """Delete the stored sessions (one prime + one fork per tick). The project dir for our private cwd holds only these files."""
        import glob
        for d in glob.glob(os.path.expanduser("~/.claude/projects/*evg-claude-cwd*")):
            for f in glob.glob(os.path.join(d, "*.jsonl")):
                try: os.remove(f)
                except OSError: pass
        self._sessions, self._primed = [], None


def _post(url, headers, body, timeout):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise ProviderError(f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}")
    except (urllib.error.URLError, TimeoutError) as e:
        raise ProviderError(f"connection failed: {e}")


class AnthropicAPI:
    def __init__(self, model="claude-haiku-4-5-20251001", api_key_env="ANTHROPIC_API_KEY", timeout=120, max_tokens=1024, effort="low", **_):
        self.model, self.timeout, self.max_tokens, self.key_env, self.effort = model, timeout, max_tokens, api_key_env, effort
        self.name, self.usage = f"anthropic:{model}", {}

    def complete(self, system, parts, schema=None):
        key = os.environ.get(self.key_env)
        if not key:
            raise ProviderError(f"{self.key_env} is not set")
        t = time.time()
        r = _post("https://api.anthropic.com/v1/messages", {"x-api-key": key, "anthropic-version": "2023-06-01"},
                  {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                   **({"output_config": {"effort": self.effort}} if self.effort and "haiku" not in self.model else {}),   # Haiku 4.5 has no effort control
                   "messages": [{"role": "user", "content": _anthropic_content(parts)}]}, self.timeout)
        self.usage = _usage(r.get("usage"), round((time.time() - t) * 1000))
        return "".join(b.get("text", "") for b in r.get("content", []))


class OpenAICompat:
    def __init__(self, model, base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY", timeout=180, max_tokens=1024, effort="low", **_):
        self.model, self.base, self.timeout, self.max_tokens, self.key_env = model, base_url.rstrip("/"), timeout, max_tokens, api_key_env
        self.effort = effort
        self.name, self.usage = f"openai:{model}@{self.base}", {}

    def complete(self, system, parts, schema=None):
        content = []
        for p in parts:
            if p is CACHE_BREAK:
                continue          # OpenAI-style servers cache the longest common prefix automatically; nothing to mark
            content.append({"type": "text", "text": p} if isinstance(p, str)
                           else {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _b64(p)}})
        key = os.environ.get(self.key_env, "")
        headers = {"authorization": f"Bearer {key}"} if key else {}
        if not key and "api.openai.com" in self.base:
            raise ProviderError(f"{self.key_env} is not set")
        t = time.time()
        r = _post(f"{self.base}/chat/completions", headers,
                  {"model": self.model, "max_tokens": self.max_tokens, "temperature": 0,
                   **({} if "api.openai.com" in self.base else {"cache_prompt": True}),      # llama.cpp: reuse the KV cache of the common prefix
                   # local thinking models (Gemma 4, Qwen 3 on Ollama) think by default: 200-850 hidden tokens per tick. effort "low" turns it off
                   **({"reasoning_effort": "none" if self.effort == "low" else self.effort} if self.effort and "api.openai.com" not in self.base else {}),
                   # local servers: constrain the output to the reply schema (valid structure, stops when the object is complete)
                   **({"response_format": {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}}}
                      if schema and "api.openai.com" not in self.base else {}),
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}, self.timeout)
        self.usage = _usage(r.get("usage"), round((time.time() - t) * 1000))
        try:
            return r["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            raise ProviderError(f"unexpected response: {str(r)[:300]}")


def make(cfg):
    """cfg: dict(preset=... | provider=, model=, base_url=, api_key_env=)."""
    cfg = {**PRESETS.get(cfg.get("preset"), {}), **{k: v for k, v in cfg.items() if v}}
    kind = cfg.get("provider")
    cls = {"claude-cli": ClaudeCLI, "anthropic": AnthropicAPI, "openai": OpenAICompat}.get(kind)
    if not cls:
        raise ProviderError(f"unknown provider {kind!r}")
    return cls(**{k: v for k, v in cfg.items() if k not in ("provider", "preset")})


def extract_json(text):
    """First balanced {...} object in text (models sometimes wrap JSON in prose or code fences)."""
    s = text.find("{")
    while s != -1:
        depth, in_str, esc = 0, False, False
        for i in range(s, len(text)):
            c = text[i]
            if in_str:
                esc = (c == "\\" and not esc)
                if c == '"' and not esc: in_str = False
            elif c == '"': in_str = True
            elif c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try: return json.loads(text[s:i + 1])
                    except ValueError: break
        s = text.find("{", s + 1)
    raise ValueError("no JSON object in model reply")
