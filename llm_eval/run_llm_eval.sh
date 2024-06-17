source secrets.env
source settings.env

cat llm_eval.sh | ssh -i ~/.ssh/tensordock -p 30118 user@$SERVER_IP # This will run the script on the remote machine

# This will copy the output file from the remote machine
sftp digisus <<EOF 
get ~/llm_eval/output/*.json
EOF

# Turn off vm
curl --location "https://marketplace.tensordock.com/api/v0/client/delete/single" \ 
--data-urlencode "api_key=$API_KEY" \
--data-urlencode "api_token=$API_TOKEN" \
--data-urlencode "server=$SERVER_ID"

# Check if vm was turned off
curl --location "https://marketplace.tensordock.com/api/v0/client/list" \
--data-urlencode "api_key=$API_KEY" \
--data-urlencode "api_token=$API_TOKEN" \