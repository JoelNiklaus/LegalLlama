# SETUP

See: https://github.com/TheElevatedOne/Tensordock-server-for-AI
Add api_key and api_token from tensordock to secrets.env. These will be used in most other commands.

Generate .ssh key-pair:

```bash
ssh-keygen -f ~/.ssh/tensordock -t rsa -b 4096
cat ~/.ssh/tensordock.pub
```

Paste output into [SSH Public Keys](https://dashboard.tensordock.com/api) to have access to servers.

for better readability of output:

```bash
sudo apt install jq # filter for json output
```

```bash
bash get_hostnodes.sh # get a list of host_nodes matching your settings
```

Paste hostnode_id into settings.env -> HOSTNODE

Start the server

```bash
bash deploy_server.sh
```

Paste server_id into settings.env -> SERVER_ID
Paste server_ip into settings.env -> SERVER_IP
