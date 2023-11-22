import csv
import os

def read_third_column(file_path):
    third_column_data = set()
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split('\t')[2]  # 假设数据以制表符分隔，并且第三列是要提取的数据
            third_column_data.add(data)
    return third_column_data

def write_to_csv(data, output_file):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for item in data:
            writer.writerow([item])

def process_files(input_folder, output_file):
    all_data = set()
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(input_folder, file_name)
            file_data = read_third_column(file_path)
            all_data.update(file_data)
    write_to_csv(all_data, output_file)

# 指定输入文件夹和输出文件路径
input_folder = 'data_compressed/WN18RR/'  # 替换为包含输入文件的文件夹路径
output_file = 'WN_output3.csv'  # 替换为输出文件的路径

# 处理文件
process_files(input_folder, output_file)
