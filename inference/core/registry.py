import importlib
import logging
from typing import Type

from inference.core.types import PluginConfig
from inference.plugins.base import CyberVersePlugin

logger = logging.getLogger(__name__)


def import_plugin_class(dotted_path: str) -> Type[CyberVersePlugin]:
    """Dynamically import a plugin class from a dotted path.

    Example: 'inference.plugins.avatar.flash_head_plugin.FlashHeadAvatarPlugin'
    """
    module_path, _, class_name = dotted_path.rpartition(".")
    if not module_path:
        raise ImportError(f"Invalid plugin class path: {dotted_path}")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not (isinstance(cls, type) and issubclass(cls, CyberVersePlugin)):
        raise TypeError(f"{dotted_path} is not a CyberVersePlugin subclass")
    return cls


class PluginRegistry:
    """Central registry for all CyberVerse plugins."""

    def __init__(self) -> None:
        self._classes: dict[str, Type[CyberVersePlugin]] = {}
        self._instances: dict[str, CyberVersePlugin] = {}

    def register(self, name: str, plugin_cls: Type[CyberVersePlugin]) -> None:
        if name in self._classes:
            raise ValueError(f"Plugin '{name}' is already registered")
        self._classes[name] = plugin_cls

    def unregister(self, name: str) -> None:
        self._classes.pop(name, None)
        self._instances.pop(name, None)

    async def initialize(self, name: str, config: PluginConfig) -> CyberVersePlugin:
        if name not in self._classes:
            raise KeyError(f"Plugin '{name}' not registered")
        instance = self._classes[name]()
        await instance.initialize(config)
        self._instances[name] = instance
        return instance

    async def initialize_all(self, configs: dict[str, PluginConfig]) -> None:
        for name, config in configs.items():
            if name in self._classes:
                await self.initialize(name, config)

    def get(self, name: str) -> CyberVersePlugin:
        if name not in self._instances:
            raise KeyError(f"Plugin '{name}' not initialized")
        return self._instances[name]

    def get_by_category(self, category: str) -> CyberVersePlugin | None:
        """Get initialized plugin by category prefix (e.g. 'avatar', 'llm')."""
        for name, instance in self._instances.items():
            if name.startswith(category + ".") or name == category:
                return instance
        return None

    def get_all_by_category(self, category: str) -> list[CyberVersePlugin]:
        """Get all initialized plugins by category prefix."""
        return [
            instance
            for name, instance in self._instances.items()
            if name.startswith(category + ".") or name == category
        ]

    async def shutdown_all(self) -> None:
        for instance in self._instances.values():
            await instance.shutdown()
        self._instances.clear()

    @property
    def registered_names(self) -> list[str]:
        return list(self._classes.keys())

    @property
    def initialized_names(self) -> list[str]:
        return list(self._instances.keys())
