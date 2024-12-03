# Unit Tests

## How to run

We use the container built following [prepare_container.md] (../docs/prepare_container.md) to run unittest.  Assuming the built container is named as `PYTORCH_CONTAINER`.  On testbed meepo4, you can use our pre-built container   `PYTORCH_CONTAINER = hjb/pytorch:v1.13.1-devel` . 

To launch the container, we can use the following command: 

```bash
cd /host/path/to/cricket
docker run -it --rm --gpus all -v /host/path/to/cricket:/testdir --privileged --network host --ipc=host $PYTORCFH_CONTAINER$ bash
cd /testdir # we are now in the cricket directory in the container
```

Then we can run the python script to launch all the tests.

```bash
python3 run_all_tests.py
```

Note: Tests involving pytorch will be slow due to kernel parsing, be patient.



We will see `pass all tests!` if all the tests run successfully.

## How to add unit tests

Each unit test must contain a `compile.sh` and a `run.sh` in its directory. The python scripts will change the working directory to the target directory and run the 2 scripts to compile and run the unit tests.
