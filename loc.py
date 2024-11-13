import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pt
import seaborn as sns
import numpy as np
import xml.etree.ElementTree as ET

# Gọi các hàm bạn đã viết
def main():
    # 1. Lấy dữ liệu từ các tệp XML
    all_test, trial_value = get_all_xml()

    # 2. Tính tổng số lượng ảnh train/test/total
    folder_name, train_counts, test_counts, total_counts = get_image_counts("Collect_File")

    # 3. Vẽ biểu đồ số lượng ảnh
    show_histogram(folder_name, train_counts, test_counts, total_counts)

    # 4. Vẽ biểu đồ bounding box xếp chồng
    plt.figure(figsize=(8, 8))
    for detect_list in all_test["detect"]:
        for obj in detect_list:
            x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
            plt.gca().add_patch(
                pt.Rectangle(
                    (x - w / 2, y - h / 2), w, h,
                    edgecolor='blue', facecolor='none', linewidth=0.5
                )
            )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Bounding Box Distribution")
    plt.xlabel("Normalized X")
    plt.ylabel("Normalized Y")
    plt.grid()
    plt.show()

    # 5. Vẽ biểu đồ mật độ (x, y)
    x_coords = []
    y_coords = []
    for detect_list in all_test["detect"]:
        for obj in detect_list:
            x_coords.append(obj["x"])
            y_coords.append(obj["y"])

    sns.kdeplot(x=x_coords, y=y_coords, cmap="Blues", fill=True, levels=100)
    plt.title("Density of Bounding Box Centers (x, y)")
    plt.xlabel("Normalized X")
    plt.ylabel("Normalized Y")
    plt.grid()
    plt.show()

    # 6. Vẽ biểu đồ mật độ width và height
    widths = []
    heights = []
    for detect_list in all_test["detect"]:
        for obj in detect_list:
            widths.append(obj["w"])
            heights.append(obj["h"])

    sns.kdeplot(x=widths, y=heights, cmap="Blues", fill=True, levels=100)
    plt.title("Density of Bounding Box Width and Height")
    plt.xlabel("Normalized Width")
    plt.ylabel("Normalized Height")
    plt.grid()
    plt.show()


# Chạy chương trình chính
if __name__ == "__main__":
    main()
