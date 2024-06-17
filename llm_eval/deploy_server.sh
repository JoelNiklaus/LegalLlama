source secrets.env
source settings.env

curl --location 'https://marketplace.tensordock.com/api/v0/client/deploy/single' \
--data-urlencode "api_key=$API_KEY" \
--data-urlencode "api_token=$API_TOKEN" \
--data-urlencode "name=$NAME" \
--data-urlencode "gpu_count=$GPU_COUNT" \
--data-urlencode "gpu_model=$GPU_MODEL" \
--data-urlencode "vcpus=$VCPUS" \
--data-urlencode "ram=$RAM" \
--data-urlencode "external_ports=$EXTERNAL_PORTS" \
--data-urlencode "internal_ports=$INTERNAL_PORTS" \
--data-urlencode "hostnode=$HOSTNODE" \
--data-urlencode "storage=$STORAGE" \
--data-urlencode "operating_system=$OPERATING_SYSTEM"