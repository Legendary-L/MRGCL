import csv


def read_first_column(csv_files):
    values_set = set()  # 用于存储去重后的值

    for file in csv_files:
        with open(file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 0:  # 确保至少有一列数据
                    value = row[0]  # 第一列的值
                    values_set.add(value)

    return values_set


def write_to_csv(values_set, output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for value in values_set:
            writer.writerow([value])


# 示例用法
csv_files = ['WN_output1.csv', 'WN_output3.csv']
output_file = 'WN_output_last.csv'
values_set = read_first_column(csv_files)
write_to_csv(values_set, output_file)