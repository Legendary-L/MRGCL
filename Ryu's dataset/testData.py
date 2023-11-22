import csv


def extract_unique_values(input_file, output_file):
    unique_values = set()  # 用于存储不重复的值

    # 读取CSV文件的第一列，记录不重复的值
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            value = row[0]  # 第一列的值
            unique_values.add(value)

    # 将不重复的值写入新的CSV文件
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for value in unique_values:
            writer.writerow([value])


# 示例用法
input_file = 'Ryu\'s\ dataset/DDl_event.csv'
output_file = 'Ryu\'s\ dataset/drug_list.csv'
extract_unique_values(input_file, output_file)
