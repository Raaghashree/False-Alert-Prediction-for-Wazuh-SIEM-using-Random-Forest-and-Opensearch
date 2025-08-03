 from opensearchpy import OpenSearch, helpers
import joblib
import time
import traceback
import json
from preprocessor import preprocess

# --- Config ---
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
AUTH = ("admin", "pMrLBp7mQdF89vQ8n70?unmXxVHVQhye")
INDEX = "wazuh-alerts-4.x-*"
MODEL_PATH = "model.pkl"
ENCODER_PATH = "data.srcip_encoder.pkl"
FEATURE_NAMES_PATH = "feature_names.pkl"

# --- Connect to OpenSearch ---
client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
    http_auth=AUTH,
    use_ssl=True,
    verify_certs=False,  # Wazuh typically uses self-signed certs
    ssl_show_warn=False
)

# --- Load model and encoders ---
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)

# --- Track last seen timestamp ---
last_seen = "now-5m"  # Start with last 5 minutes

print("[*] Starting alert classifier enrichment...")

# --- Loop forever ---
while True:
    try:
        response = client.search(
            index=INDEX,
            body={
                "size": 100,
                "query": {
                    "range": {
                        "@timestamp": {
                            "gt": last_seen
                        }
                    }
                },
                "sort": [
                    {"@timestamp": "asc"}
                ]
            }
        )

        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            print("[-] No new alerts.")
        else:
            print(f"[+] Processing {len(hits)} new alerts...")

        actions = []
        latest_timestamp = last_seen  # Track latest valid timestamp

        for hit in hits:
            try:
                doc = hit["_source"]
                doc_id = hit["_id"]
                index = hit["_index"]

                # --- Preprocess features ---
                features = preprocess(doc)

                # --- Predict probability ---
                prob = model.predict_proba([features])[0][1]
                prob_percent = round(prob * 100, 2)

                # --- Build update action ---
                actions.append({
                    "_op_type": "update",
                    "_index": index,
                    "_id": doc_id,
                    "doc": {
                        "ml_false_prediction": f"{prob_percent}%"
                    }
                })

                # Safely update timestamp only if valid
                if "@timestamp" in doc:
                    latest_timestamp = doc["@timestamp"]

            except Exception as e:
                print(f"[!] Error processing document {hit.get('_id')}: {e}")
                traceback.print_exc()

        # --- Bulk update ---
        if actions:
            print("[*] Sample update action:")
            print(json.dumps(actions[-1], indent=2))

            helpers.bulk(client, actions)
            print(f"[✓] Updated {len(actions)} alerts with ML predictions.")
            last_seen = latest_timestamp

    except Exception as e:
        print(f"[!] Error querying OpenSearch: {e}")
        traceback.print_exc()

    time.sleep(10)
