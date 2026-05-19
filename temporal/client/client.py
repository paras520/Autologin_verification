from __future__ import annotations

from temporalio.client import Client, TLSConfig

from temporal.config.settings import TEMPORAL_API_KEY, TEMPORAL_HOST, TEMPORAL_NAMESPACE


async def get_temporal_client() -> Client:
    """Return a connected Temporal client.

    For local dev (no API key) a plain unencrypted connection is used.
    For Temporal Cloud, pass TEMPORAL_API_KEY; TLS is enabled automatically.
    """
    if TEMPORAL_API_KEY:
        return await Client.connect(
            TEMPORAL_HOST,
            namespace=TEMPORAL_NAMESPACE,
            tls=TLSConfig(
                server_root_ca_cert=None,
                client_cert=None,
                client_private_key=None,
            ),
            rpc_metadata={"temporal-namespace": TEMPORAL_NAMESPACE},
            api_key=TEMPORAL_API_KEY,
        )

    return await Client.connect(
        TEMPORAL_HOST,
        namespace=TEMPORAL_NAMESPACE,
    )
