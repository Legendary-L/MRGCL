import csv

def read_third_column(csv_files):
    values_set = set()  # 用于存储去重后的值
    
    for file in csv_files:
        with open(file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 2:  # 确保至少有三列数据
                    value = row[2]  # 第三列的值
                    values_set.add(value)
    
    return values_set

def write_to_csv(values_set, output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for value in values_set:
            writer.writerow([value])


# 示例用法
csv_files = ['data/0/ddi_test1.csv', 'data/0/ddi_training1.csv', 'data/0/ddi_validation1.csv', 
'data/1/ddi_test1.csv', 'data/1/ddi_training1.csv', 'data/1/ddi_validation1.csv', 
'data/2/ddi_test1.csv', 'data/2/ddi_training1.csv', 'data/2/ddi_validation1.csv', 
'data/3/ddi_test1.csv', 'data/3/ddi_training1.csv', 'data/3/ddi_validation1.csv', 
'data/4/ddi_test1.csv', 'data/4/ddi_training1.csv', 'data/4/ddi_validation1.csv']

output_file = 'data/output3.csv'
values_set = read_third_column(csv_files)
write_to_csv(values_set, output_file)
