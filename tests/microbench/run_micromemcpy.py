import os
import glob
import subprocess
import signal
import psutil
import sys
import time

KB = 1024
MB = 1024 * 1024

VERSION = os.environ.get('VERSION')
if VERSION is None:
    print('VERSION is not set!')
    sys.exit()

files = [
    'cuda_app/cudaMemcpyAsyncH2D.out'
]

memory_sizes = [
    128 * KB,
    256 * KB,
    512 * KB,
    1 * MB,
    2 * MB,
    4 * MB,
    8 * MB,
    16 * MB,
    32 * MB,
    64 * MB
]

available_versions = ['WITH_VANILLA', 'NO_OPTIMIZATION', 'WITH_TCPIP', 'WITH_RDMA', 'WITH_SHARED_MEMORY']
if VERSION not in available_versions:
    print('VERSION must in ', available_versions)
    sys.exit()

def start_server(file_name):
    process = subprocess.Popen(['bash', 'startserver.sh'], stdout=open(file_name, 'w'), bufsize=0)
    return process.pid

def kill_server(pid):
    time.sleep(10)
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass

def compile():
    print('start to compile cricket!')
    current_directory = os.getcwd()
    os.chdir('../..')
    # compile_command = 'make -C cpu clean && make cpu VERSION=%s NO_CACHE_OPTIMIZATION=1 NO_ASYNC_OPTIMIZATION=1 NO_HANDLER_OPTIMIZATION=1 -j' % (VERSION)
    compile_command = 'make -C cpu clean && make cpu VERSION=%s -j' % (VERSION)
    make_result = os.system(compile_command)
    if make_result == 0:
        print('successfully build cricket!')
        os.chdir(current_directory)
    else:
        print('fail to build cricket!')
        os._exit(-1)

def run_microbenchmark(executable, test_name, version, result_dir):
    for memory_size in memory_sizes:
        current_test_name = test_name + '_' + str(memory_size)
        if version == 'WITH_VANILLA':
            result_file = '%s/%s_%s.txt' % (result_dir, current_test_name, version)
            print('running microbenchmark %s' % (current_test_name))
            run_command = 'MEMORY_SIZE=%s bash runvanilla.sh %s > %s' % (str(memory_size), executable, result_file)
            run_result = os.system(run_command)
        else:
            result_client_file = '%s/%s_%s_client.txt' % (result_dir, current_test_name, version)
            result_server_file = '%s/%s_%s_server.txt' % (result_dir, current_test_name, version)

            server_pid = start_server(result_server_file)
            print('running microbenchmark %s, server pid: %d' % (current_test_name, server_pid))

            run_command = 'MEMORY_SIZE=%s bash startclient.sh %s > %s' % (str(memory_size), executable, result_client_file)
            time.sleep(3)
            run_result = os.system(run_command)
            kill_server(server_pid)

        if run_result != 0:
            print('fail to run microbenchmark %s' % (executable))

if __name__ == '__main__':
    compile()
    result_dir = 'logs'
    os.makedirs(result_dir, exist_ok=True)
    for f in files:
        executable = f
        filename = f.split('/')[-1]
        start_index = filename.find('*')
        end_index = filename.rfind('.out')
        test_name = filename[start_index + 1: end_index]
        run_microbenchmark(executable, test_name, VERSION, result_dir)
