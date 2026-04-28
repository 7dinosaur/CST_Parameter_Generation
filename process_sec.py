import numpy as np
import os

def process_sec(origin_folder, new_folder='processed_sec'):
    """遍历文件夹下的tecplot文件，提取坐标对，排序去重，写入新文件夹"""
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    
    for filename in os.listdir(origin_folder):
        if filename.endswith('.dat'):  # 假设tecplot文件是.dat
            filepath = os.path.join(origin_folder, filename)
            coordinates = []
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # 找到数据开始的行（跳过表头）
            data_start = False
            for line in lines:
                line = line.strip()
                if line.startswith('DT='):  # 表头结束，数据开始
                    data_start = True
                    continue
                if data_start:
                    if 'E' in line or '.' in line:  # 坐标行有浮点数
                        parts = line.split()
                        if len(parts) == 2:
                            try:
                                x = float(parts[0])
                                y = float(parts[1])
                                coordinates.append((x, y))
                            except ValueError:
                                continue
                    # 跳过索引行
            
            # 排序并去重
            coordinates.sort(key=lambda p: p[0])  # 按x排序
            unique_coordinates = []
            seen = set()
            for coord in coordinates:
                coord_tuple = (round(coord[0], 10), round(coord[1], 10))  # 精度控制去重
                if coord_tuple not in seen:
                    seen.add(coord_tuple)
                    unique_coordinates.append(coord)
            
            # 区分上下表面
            if len(unique_coordinates) < 2:
                final_coords = unique_coordinates
            else:
                point_min = unique_coordinates[0]
                point_max = unique_coordinates[-1]
                x_min, y_min = point_min
                x_max, y_max = point_max
                if x_max == x_min:
                    m = 0
                else:
                    m = (y_max - y_min) / (x_max - x_min)
                
                def y_line(x):
                    return m * (x - x_min) + y_min
                
                lower = []
                upper = []
                for coord in unique_coordinates:
                    x, y = coord
                    yl = y_line(x)
                    if y > yl:
                        upper.append(coord)
                    else:
                        lower.append(coord)
                
                # 确保头尾在对应表面
                if point_min not in lower:
                    lower.append(point_min)
                if point_max not in upper:
                    upper.append(point_max)
                
                # 去重并排序
                lower = list(set(lower))
                upper = list(set(upper))
                lower.sort(key=lambda p: p[0])
                upper.sort(key=lambda p: p[0])
                
                # 下表面反序
                lower.reverse()
                
                # 合并：下表面反序 + 上表面正序
                final_coords = lower + upper
            
            # 写入新文件
            new_filepath = os.path.join(new_folder, filename)
            with open(new_filepath, 'w') as f:
                for x, y in final_coords:
                    f.write(f"{x:.10e} {y:.10e}\n")

if __name__ == "__main__":
    process_sec('new_sbwb_sec')