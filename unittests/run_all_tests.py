import os
import glob
import subprocess
import signal
import psutil

failed_tests = []

VERSION = os.environ.get('VERSION')
if VERSION is None:
    VERSION = 'NO_OPTIMIZATION'

def start_server():
    process = subprocess.Popen(['bash', 'startserver.sh'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process.pid

def kill_server(pid):
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
    os.chdir('..')
    compile_command = 'make -C cpu clean && make cpu VERSION=%s' % (VERSION)
    make_result = os.system(compile_command)
    if make_result == 0:
        print('successfully build cricket!')
        os.chdir(current_directory)
    else:
        print('fail to build cricket!')
        os._exit(-1)

def find_unittests():
    print('finding unit tests in current directory...')
    results = []
    current_directory = os.getcwd()
    subdirectories = glob.glob(os.path.join(current_directory, '*'))
    for directory in subdirectories:
        if os.path.isfile(os.path.join(directory, 'compile.sh')) and os.path.isfile(os.path.join(directory, 'run.sh')):
            print('find unit test %s' % directory)
            results.append(directory)
        else:
            print('ignore %s' % directory)
    return results

import time

def run_unittests(tests):
    print('start running unit tests!')
    upper_directory = os.getcwd()
    for test in tests:
        os.chdir(upper_directory)
        print('=======================')
        server_pid = start_server()
        print('run unit test in %s, server pid: %d' % (test, server_pid))
        time.sleep(1)
        os.chdir(test)
        compile_result = os.system('bash compile.sh')
        if compile_result != 0:
            print('fail to compile test in %s' % test)
            failed_tests.append(test)
            kill_server(server_pid)
            continue
        run_result = os.system('bash run.sh')
        if run_result != 0:
            print('fail to run test in %s' % test)
            failed_tests.append(test)
            kill_server(server_pid)
            continue

if __name__ == '__main__':
    compile()
    tests = find_unittests()
    run_unittests(tests)
    if failed_tests:
        print('failed tests: ', failed_tests)
    else:
        print('pass all tests!')
