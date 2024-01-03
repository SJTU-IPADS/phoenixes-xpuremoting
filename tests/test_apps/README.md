# Run RestAPPs

## through RDMA

1. On both node `server` and node `client`:

```bash
cd path/to/cricket
VERSION=WITH_RDMA make cpu -j
cd tests/test_apps
```

2. On node `server`:

```bash
XPU_REMOTE_ADDRESS=$client ./startserver.sh
```

3. On node `client`:

```bash
XPU_REMOTE_ADDRESS=$server ./startinference.sh $app $num_iter $batch_size
```

For instance, app can be `GPT2/inference.py`.
