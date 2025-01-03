class AnthropicInstrumentor(BaseInstrumentor):
    def __init__(self):
        super().__init__()
        self.use_legacy_attributes = True
        
    def _instrument(self, **kwargs):
        self.use_legacy_attributes = kwargs.get('use_legacy_attributes', True)
        # ... rest of instrumentation 