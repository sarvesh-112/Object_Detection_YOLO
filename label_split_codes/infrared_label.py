import os
import shutil

def collect_infrared_labels(split_label_dir, output_label_dir):
    os.makedirs(output_label_dir, exist_ok=True)
    count = 0

    for phase in ["train", "val"]:
        phase_dir = os.path.join(split_label_dir, phase)
        for file in os.listdir(phase_dir):
            if file.endswith(".txt"):
                src = os.path.join(phase_dir, file)
                dst = os.path.join(output_label_dir, file)
                shutil.copy(src, dst)
                count += 1

    print(f" Collected {count} label files into '{output_label_dir}'")

if __name__ == "__main__":
    split_label_dir = "C:/Users/smpga/PycharmProjects/Object_detection/datasets_split/infrared/labels"
    output_label_dir = "C:/Users/smpga/PycharmProjects/Object_detection/labels/infrared"

    collect_infrared_labels(split_label_dir, output_label_dir)
