import os

# Paths to label folders
teacher_labels = 'SCB5-Teacher-2024-9-17/labels'
student_labels = 'SCB5-Handrise-Read-write-2024-9-17/labels'
combined_labels = 'combined_dataset/labels'

def relabel_files(label_dir, output_dir, new_class_id):
    for split in ['train', 'val']:
        input_path = os.path.join(label_dir, split)
        output_path = os.path.join(output_dir, split)
        os.makedirs(output_path, exist_ok=True)

        for file in os.listdir(input_path):
            if file.endswith('.txt'):
                with open(os.path.join(input_path, file), 'r') as f:
                    lines = f.readlines()
                new_lines = [f"{new_class_id} " + line.split(' ', 1)[1] for line in lines]
                with open(os.path.join(output_path, file), 'w') as f:
                    f.writelines(new_lines)

# Relabel teacher data as class 0
relabel_files(teacher_labels, combined_labels, new_class_id=0)

# Relabel student data (handrise, read, write) as class 1
relabel_files(student_labels, combined_labels, new_class_id=1)
