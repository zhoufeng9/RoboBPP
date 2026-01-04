import numpy as np
import matplotlib.pyplot as plt

# 定义图像大小和正方形大小
image_size = (2000, 2000, 3)
square_size = 1500

# 创建一个黑色背景
image = np.zeros(image_size, dtype=np.uint8)

# 计算放置白色正方形的位置
start_x = (image_size[0] - square_size) // 2
start_y = (image_size[1] - square_size) // 2

# 在中心位置放置白色正方形
image[start_x:start_x+square_size, start_y:start_y+square_size] = 255

# 保存图像
output_path = '4.png'
plt.imsave(output_path, image)

# 显示生成的图像
plt.imshow(image)
plt.axis('off')
plt.show()
