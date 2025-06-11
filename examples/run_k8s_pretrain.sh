#!/bin/bash

API_URL=""
CMD=""
WORKLOAD_ID=""

REPLICA=1
CPU="192"
GPU="8"
EXP_PATH=""
DATA_PATH=""
IMAGE="docker.io/rocm/megatron-lm:v25.5_py310"
HF_TOKEN="${HF_TOKEN:-}"

usage() {
    cat <<EOF
Usage: $0 --url <api_base_url> <command> [options]

Commands:
    create                      Create a workload (using inline JSON payload)
    get --workload-id <id>      Get workload details
    delete --workload-id <id>   Delete a workload
    list                        List all workloads

Options for create:
    --replica <num>             Number of replicas (default: 1)
    --cpu <cpu_count>           CPU count (default: 192)
    --gpu <gpu_count>           GPU count (default: 8)
    --exp <exp_path>            Path to EXP config (optional)
    --data_path <data_path>     Data path (optional)
    --image <docker_image>      Docker image to use (default: docker.io/rocm/megatron-lm:v25.5_py310)
    --hf_token <token>          HuggingFace token (default: from env HF_TOKEN)

Other:
    --help                      Show this help message

Examples:

    # Create a workload with custom resources and paths
    $0 --url http://api.example.com create --replica 2 --cpu 96 --gpu 4 \\
        --exp examples/megatron/configs/llama2_7B-pretrain.yaml --data_path /mnt/data/train \\
        --image docker.io/custom/image:latest --hf_token myhf_token

    # Get workload details
    $0 --url http://api.example.com get --workload-id abc123

    # Delete a workload
    $0 --url http://api.example.com delete --workload-id abc123

    # List all workloads
    $0 --url http://api.example.com list

EOF
    exit 1
}

if ! command -v jq >/dev/null 2>&1; then
    echo "Error: jq is required but not installed."
    exit 1
fi

if [ $# -lt 2 ]; then
    usage
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --url)
            API_URL="$2"
            shift 2
            ;;
        create|get|delete|list)
            CMD="$1"
            shift
            ;;
        --workload-id)
            WORKLOAD_ID="$2"
            shift 2
            ;;
        --replica)
            REPLICA="$2"
            shift 2
            ;;
        --cpu)
            CPU="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --exp)
            EXP_PATH="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --hf_token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown param: $1"
            usage
            ;;
    esac
done

if [[ -z "$API_URL" || -z "$CMD" ]]; then
    usage
fi

if [[ "$CMD" == "create" && -z "$EXP_PATH" ]]; then
    echo "Error: --exp <exp_path> is required for create command."
    exit 1
fi

USER_NAME=$(whoami)
CUR_DIR=$(pwd)

ENV_JSON="{}"

if [ -n "$HF_TOKEN" ]; then
    ENV_JSON=$(echo "$ENV_JSON" | jq --arg hf "$HF_TOKEN" '. + {HF_TOKEN: $hf}')
fi
if [ -n "$EXP_PATH" ]; then
    ENV_JSON=$(echo "$ENV_JSON" | jq --arg exp "$EXP_PATH" '. + {EXP: $exp}')
fi
if [ -n "$DATA_PATH" ]; then
    ENV_JSON=$(echo "$ENV_JSON" | jq --arg data "$DATA_PATH" '. + {DATA_PATH: $data}')
fi

ENTRY_POINT="cd $CUR_DIR; bash ./examples/run_pretrain.sh 1>output/\$WORKLOAD_ID.k8s-job.log 2>&1"

read -r -d '' INLINE_JSON <<EOF || true
{
    "workspace": "safe-cluster-dev",
    "displayName": "test-primus",
    "groupVersionKind": {
        "kind": "PyTorchJob",
        "group": "kubeflow.org",
        "version": "v1"
    },
    "description": "test primus",
    "userName": "$USER_NAME",
    "entryPoint": "$ENTRY_POINT",
    "isSupervised": false,
    "image": "$IMAGE",
    "maxRetry": 1,
    "resource": {
        "replica": $REPLICA,
        "cpu": "$CPU",
        "gpu": "$GPU",
        "memory": "1024Gi",
        "ephemeralStorage": "100Gi"
    },
    "env": $ENV_JSON
}
EOF

curl_post() {
    curl --fail -s -H "Content-Type: application/json" -X POST -d "$INLINE_JSON" "$API_URL/api/v1/workloads"
}

curl_get() {
    curl --fail -s "$API_URL/api/v1/workloads/$1"
}

curl_delete() {
    curl --fail -s -X DELETE "$API_URL/api/v1/workloads/$1"
}

curl_list() {
    curl --fail -s "$API_URL/api/v1/workloads"
}

case "$CMD" in
    create)
        echo "Creating workload with inline JSON..."
        RESPONSE=$(curl_post) || { echo "Create failed"; exit 1; }
        echo "$RESPONSE" | jq .
        ;;
    get)
        if [ -z "$WORKLOAD_ID" ]; then
            echo "Missing --workload-id for get"
            exit 1
        fi
        RESPONSE=$(curl_get "$WORKLOAD_ID") || { echo "Get failed"; exit 1; }
        echo "$RESPONSE" | jq .
        ;;
    delete)
        if [ -z "$WORKLOAD_ID" ]; then
            echo "Missing --workload-id for delete"
            exit 1
        fi
        RESPONSE=$(curl_delete "$WORKLOAD_ID") || { echo "Delete failed"; exit 1; }
        echo "$RESPONSE" | jq .
        ;;
    list)
        RESPONSE=$(curl_list) || { echo "List failed"; exit 1; }
        echo "$RESPONSE" | jq .
        ;;
    *)
        usage
        ;;
esac
