# DBs/azure_cosmos.py
import os
import uuid
from dotenv import load_dotenv

load_dotenv(override=True)

def log_to_cosmos(user_input: str, reports: list):
    try:
        from azure.cosmos import CosmosClient, PartitionKey

        url = os.getenv("COSMOS_ENDPOINT")
        key = os.getenv("COSMOS_KEY")

        if not url or not key:
            print("⚠️  [Cosmos] No credentials — skipping log.")
            return

        client   = CosmosClient(url, key)
        database = client.create_database_if_not_exists(id="BMS")

        # ✅ FIX: بدون throughput عشان serverless accounts
        try:
            container = database.create_container_if_not_exists(
                id="Logs",
                partition_key=PartitionKey(path="/user_query")
            )
        except Exception:
            container = database.get_container_client("Logs")

        container.upsert_item({
            "id":           str(uuid.uuid4()),
            "user_query":   user_input,
            "agent_report": reports
        })

        print("💾 [Cosmos] Log saved.")

    except Exception as e:
        print(f"⚠️  [Cosmos Warning] {e}")
