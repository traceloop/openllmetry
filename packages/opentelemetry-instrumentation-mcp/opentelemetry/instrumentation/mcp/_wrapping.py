"""Own MCP method patches, including references copied by importing libraries."""

from wrapt import FunctionWrapper, register_post_import_hook, resolve_path


class _InstrumentedMethods:
    def __init__(self):
        self._active = False
        self._patches = {}

    def wrap(self, module_name, name, wrapper):
        self._active = True
        key = (module_name, name)
        if key not in self._patches:
            self._patches[key] = _MethodPatch(self, name)
        patch = self._patches[key]
        # Copied references dispatch through the current callback/tracer too.
        patch.wrapper = wrapper
        if not patch.pending:
            patch.pending = True
            register_post_import_hook(patch.on_import, module_name)

    def unwrap(self):
        # Restoring attributes alone cannot recall copied or bound references.
        self._active = False
        for patch in reversed(list(self._patches.values())):
            patch.remove()
            patch.wrapper = None


class _MethodPatch:
    def __init__(self, methods, name):
        self.methods = methods
        self.name = name
        self.wrapper = None
        self.pending = False
        self.parent = None
        self.attribute = None
        self.replacement = None

    def _dispatch(self, wrapped, instance, args, kwargs):
        return self.wrapper(wrapped, instance, args, kwargs)

    def on_import(self, module):
        self.pending = False
        # A pending hook can fire while disabled. Later activation registers it
        # again, or updates the single still-pending hook's current callback.
        if not self.methods._active:
            return
        self.parent, self.attribute, original = resolve_path(module, self.name)
        for _, current in self._chain(original):
            if current is self.replacement:
                return
        if self.replacement is None or self.replacement.__wrapped__ is not original:
            # Never redirect a copied callable to a different function. An
            # external decorator may even close over our old wrapper without
            # exposing __wrapped__; retargeting it would create a call cycle.
            replacement = FunctionWrapper(
                original,
                self._dispatch,
                enabled=lambda: self.methods._active and self.replacement is replacement,
            )
            # If the target changed, retire the old wrapper. It must pass
            # through when an opaque decorator calls it underneath the new one.
            self.replacement = replacement
        setattr(self.parent, self.attribute, self.replacement)

    @staticmethod
    def _chain(current):
        previous = None
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            yield previous, current
            previous, current = current, getattr(current, "__wrapped__", None)

    def remove(self):
        if self.parent is None:
            return
        try:
            _, _, current = resolve_path(self.parent, self.attribute)
        except AttributeError:
            # Another owner can remove a target; don't resurrect it or skip
            # teardown of the remaining patches.
            return
        for previous, current in self._chain(current):
            if current is self.replacement:
                if previous is None:
                    setattr(self.parent, self.attribute, current.__wrapped__)
                elif isinstance(previous, FunctionWrapper):
                    previous.__wrapped__ = current.__wrapped__
                # An ordinary decorator's closure cannot be rewritten through
                # __wrapped__. Leave our disabled layer underneath it instead.
                break
