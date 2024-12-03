echo "VERSION=WITH_VANILLA python run_microbenchmarks.py"
VERSION=WITH_VANILLA python run_microbenchmarks.py

echo "VERSION=WITH_SHARED_MEMORY python run_microbenchmarks.py"
VERSION=WITH_SHARED_MEMORY python run_microbenchmarks.py

echo "VERSION=WITH_VANILLA python run_micromemcpy.py"
VERSION=WITH_VANILLA python run_micromemcpy.py

echo "VERSION=WITH_SHARED_MEMORY python run_micromemcpy.py"
VERSION=WITH_SHARED_MEMORY python run_micromemcpy.py
