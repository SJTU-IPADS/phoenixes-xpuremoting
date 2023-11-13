import os
import glob
import subprocess
import signal
import psutil
import sys

VERSION = os.environ.get('VERSION')
if VERSION is None:
    print('VERSION is not set!')
    sys.exit()
    
INFERENCE = os.environ.get('INFERENCE')
if INFERENCE is None:
    print('INFERENCE is not set!')
    sys.exit()

def start_server(file_name):
    process = subprocess.Popen(['bash', 'startserver.sh'], stdout=open(file_name, 'w'), bufsize=0)
    return process.pid

def kill_server(pid):
    time.sleep(60)
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
    compile_command = 'make -C cpu clean && make cpu VERSION=%s -j' % (VERSION)
    make_result = os.system(compile_command)
    if make_result == 0:
        print('successfully build cricket!')
        os.chdir(current_directory)
    else:
        print('fail to build cricket!')
        os._exit(-1)

import time

def run_inference(inference_case, iter_num, result_dir):
    # "./startinference.sh inference_case iter_num"
    inference_file = inference_case + '/inference.py'
    result_client_file = '%s/%d_iter_client.txt' % (result_dir, iter_num)
    result_server_file = '%s/%d_iter_server.txt' % (result_dir, iter_num)
    
    server_pid = start_server(result_server_file)
    print('running inference %s with %d iterations, server pid: %d' % (inference_case, iter_num, server_pid))
    
    run_command = 'bash startinference.sh %s %d > %s' % (inference_file, iter_num, result_client_file)
    time.sleep(3)
    run_result = os.system(run_command)
    if run_result != 0:
        print('fail to run')
    kill_server(server_pid)

if __name__ == '__main__':
    compile()
    iter_nums = [1000]
    result_dir = INFERENCE + '/results/' + VERSION
    os.makedirs(result_dir, exist_ok=True)
    for iter_num in iter_nums:
        run_inference(INFERENCE, iter_num, result_dir)
