source secrets.env
source settings.env
curl --location --request GET 'https://marketplace.tensordock.com/api/v0/client/deploy/hostnodes' \
--data-urlencode "minRAM=$RAM" \
--data-urlencode "minVCPUs=$VCPUS" \
--data-urlencode "minGPUs=$GPU_COUNT" \
--data-urlencode "minStorage=$STORAGE" \
| jq '.hostnodes | to_entries[] | {hostnode_id: .key, gpu: .value.specs.gpu, status: .value.status}' | grep -B 2 -A 15 $GPU_MODEL
