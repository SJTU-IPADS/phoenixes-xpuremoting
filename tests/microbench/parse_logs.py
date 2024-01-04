import glob
import os
import re
import pandas as pd

log_directory = 'logs-v100-opt-large-buffer'
excel_name = 'test.xlsx'

pattern = log_directory + '/*_WITH_VANILLA.txt'
file_names = glob.iglob(pattern)
test_names = []
for file_name in file_names:
    base_name = os.path.basename(file_name)
    string = base_name.replace('_WITH_VANILLA.txt', '')
    test_names.append(string)


def extract_elapsed_time(file_name):
    elapsed_times = []
    pattern = "Average elapsed time:"
    with open(file_name, "r") as file:
        for line in file:
            if pattern in line:
                start_index = line.find(pattern) + len(pattern) + 1
                end_index = line.find(" ms", start_index)
                elapsed_time_str = line[start_index:end_index]
                elapsed_time_float = float(elapsed_time_str)
                elapsed_times.append(elapsed_time_float)
    assert(len(elapsed_times) == 1)
    return elapsed_times[0]

def parse_vanilla(test_name):
    file_name = log_directory + '/' + test_name + '_WITH_VANILLA.txt'
    return {
        'raw api time (us)': extract_elapsed_time(file_name) * 1000
    }

def parse_client_side_log(test_name):
    file_name = log_directory + '/' + test_name + '_WITH_SHARED_MEMORY_client.txt'
    total_time_us = extract_elapsed_time(file_name) * 1000
    pattern = r'api (.+): count (.+), payload_size (.+), total_time (.+), serialization_time (.+), network_send_time (.+), network_receive_time (.+)'
    with open(file_name, 'r') as file:
        for line in file:
            match = re.match(pattern, line)
            if match:
                api = float(match.group(1))
                count = float(match.group(2))
                payload_size = float(match.group(3))
                total_time = float(match.group(4))
                serialization_time = float(match.group(5))
                network_send_time = float(match.group(6))
                network_receive_time = float(match.group(7))

                total_time = serialization_time + network_send_time + network_receive_time

                if count < 100:
                    continue

                # print(f"API: {api}")
                # print(f"Count: {count}")
                # print(f"Payload Size: {payload_size}")
                # print(f"Total Time: {total_time}")
                # print(f"Serialization Time: {serialization_time}")
                # print(f"Network Send Time: {network_send_time}")
                # print(f"Network Receive Time: {network_receive_time}")
                # # print(f"Other Time Ratio: {(total_time-serialization_time-network_send_time-network_receive_time)/total_time}")
                # print()

                return {
                    "serialization (us)": total_time_us / total_time * serialization_time,
                    "send request (us)": total_time_us / total_time * network_send_time,
                    "wait reply (us)": total_time_us / total_time * network_receive_time,
                }
    print("file name %s does not contain valid api profiling, wait reply will take up all time" % (file_name))
    return {
        "serialization (us)": 0,
        "send request (us)": 0,
        "wait reply (us)": total_time_us,
    }

if __name__ == '__main__':
    all_apis = {}
    for test_name in test_names:
        api = {}
        api['raw api time (us)'] = parse_vanilla(test_name)['raw api time (us)']
        api.update(parse_client_side_log(test_name))
        all_apis[test_name] = api
    df = pd.DataFrame(all_apis)
    df =df.T
    df.sort_index(inplace=True)
    with pd.ExcelWriter(excel_name) as writer:
        df.to_excel(writer)
