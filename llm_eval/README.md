# SETUP

See: https://github.com/TheElevatedOne/Tensordock-server-for-AI
Add api_key and api_token from tensordock to secrets.env. These will be used in most other commands.
Change settings.env to match needs.

Generate .ssh key-pair for access to the server.

```bash
ssh-keygen -f ~/.ssh/tensordock -t rsa -b 4096
cat ~/.ssh/tensordock.pub
```

Paste output into [SSH Public Keys](https://dashboard.tensordock.com/api) to have access to servers. The keys need to be comma-separated.

## Get available hostnodes

for better readability of output:

```bash
sudo apt install jq # filter for json output
```

```bash
bash get_hostnodes.sh # get a list of host_nodes matching your settings
```

Paste best hostnode_id into settings.env -> HOSTNODE

## Start the server

```bash
bash deploy_server.sh
```

Paste server_id from response into settings.env -> SERVER_ID
Paste server_ip from response into settings.env -> SERVER_IP

## Run evaluation

```bash
bash run_llm_eval.sh
```

Output of evaluation will be saved into the current directory.
