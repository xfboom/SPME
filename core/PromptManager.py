try:
    import yaml
except ModuleNotFoundError:
    yaml = None

import time
import re
from pathlib import Path
try:
    from jinja2 import Template
except ModuleNotFoundError:
    Template = None
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PromptVersion:
    """Represents a single prompt version."""
    template: str
    defaults: Dict[str, str] = field(default_factory=dict)


@dataclass
class Prompt:
    """Represents a full prompt with multiple versions."""
    name: str
    versions: Dict[str, PromptVersion]


class PromptManager:
    """
    Loads and manages prompt templates from YAML files.
    Supports versioning, default values, and Jinja2 templating.
    """

    def __init__(self, prompt_dir: str = "_prompts", hot_reload: bool = False):
        self.prompt_dir = Path(prompt_dir)
        self.hot_reload = hot_reload
        self._cache: Dict[str, Prompt] = {}
        self._last_load_time = 0
        self.reload_interval = 5
        self._load_all()

    def _load_all(self):
        self._cache.clear()
        for file in self.prompt_dir.glob("*.yaml"):
            data = self._load_prompt_file(file)
            # Supports either the simple repo format or a versioned prompt format.
            if "versions" in data:
                versions = {name: PromptVersion(**v) for name, v in data["versions"].items()}
            else:
                versions = {"v1": PromptVersion(template=data["system"])}
            prompt = Prompt(name=data["name"], versions=versions)
            self._cache[prompt.name] = prompt
        self._last_load_time = time.time()

    @staticmethod
    def _load_prompt_file(file: Path) -> dict:
        text = file.read_text(encoding="utf-8-sig")
        if yaml is not None:
            return yaml.safe_load(text)
        return PromptManager._parse_simple_prompt_yaml(text)

    @staticmethod
    def _parse_simple_prompt_yaml(text: str) -> dict:
        """Fallback parser for this repo's simple `name` + `system: |` prompt files."""
        lines = text.splitlines()
        name = None
        system_lines = []
        in_system = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("name:") and name is None:
                name = stripped.split(":", 1)[1].strip().strip("\"'")
                continue
            if stripped == "system: |":
                in_system = True
                continue
            if in_system:
                if line.startswith("  "):
                    system_lines.append(line[2:])
                elif not stripped:
                    system_lines.append("")
                else:
                    in_system = False

        if not name or not system_lines:
            raise ValueError(
                "PyYAML is not installed and the fallback parser only supports prompt files "
                "with top-level `name` and block-style `system: |` fields."
            )

        return {"name": name, "system": "\n".join(system_lines).rstrip()}

    def _maybe_reload(self):
        if not self.hot_reload:
            return
        if time.time() - self._last_load_time > self.reload_interval:
            self._load_all()

    def get(self, name: str, version: str = "v1") -> PromptVersion:
        self._maybe_reload()
        prompt = self._cache.get(name)
        if not prompt:
            raise KeyError(f"Prompt '{name}' not found")
        if version not in prompt.versions:
            raise KeyError(f"Version '{version}' not found")
        return prompt.versions[version]

    def render(self, name: str, version: str = "v1", **kwargs) -> str:
        prompt_version = self.get(name, version)
        vars = {**prompt_version.defaults, **kwargs}
        if Template is not None:
            template = Template(prompt_version.template)
            return template.render(**vars)
        return self._render_simple_template(prompt_version.template, vars)

    @staticmethod
    def _render_simple_template(template: str, vars: dict) -> str:
        def replace_join(match):
            key = match.group(1).strip()
            sep = match.group(2).encode("utf-8").decode("unicode_escape")
            value = vars.get(key, [])
            if isinstance(value, (list, tuple)):
                return sep.join(str(item) for item in value)
            return str(value)

        rendered = re.sub(
            r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\|\s*join\(\"(.*?)\"\)\s*\}\}",
            replace_join,
            template,
        )

        def replace_var(match):
            key = match.group(1).strip()
            return str(vars.get(key, ""))

        return re.sub(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}", replace_var, rendered)
