import os
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
import weaviate


from traceloop.sdk import Traceloop

# Set this to True for first run
create_data = False
log_to_stdout = True
kwargs = {
    "app_name": "weviate_st_app",
    "disable_batch": True,
}
if log_to_stdout:
    kwargs["exporter"] = ConsoleSpanExporter()

# Init trace
Traceloop.init(**kwargs)

auth_config = weaviate.AuthApiKey(api_key=os.environ["WEAVIATE_API_KEY"])

client = weaviate.Client(
  url=os.environ["WEAVIATE_CLUSTER_URL"],
  auth_client_secret=auth_config
)
