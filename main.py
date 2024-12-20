from infer import detect
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from scipy.ndimage import label
import cv2
from ultralytics import YOLO


def show_combined_mask(combined_mask):
    """
    Show the combined mask using matplotlib.
    
    Args:
        combined_mask (np.array): Combined mask array.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(np.max(combined_mask, axis=2), cmap='gray')
    plt.title("Combined Mask")
    plt.axis('off')
    plt.show()

def get_mask_coordinates(mask):
    """
    Get coordinates of all regions in the binary mask.
    
    Args:
        mask (np.array): Binary mask.
    
    Returns:
        list: List of coordinates for each region.
    """
    labeled_mask, num_features = label(mask)
    regions = regionprops(labeled_mask)
    all_coords = [region.coords for region in regions]
    return all_coords

def is_point_in_polygon(results, approx):
    res=[]
    for result in results:
        x1, y1, x2, y2, center_point = result
        # 判断中心点是否在四边形内
        inside = cv2.pointPolygonTest(approx, center_point, False)
        if inside >= 0:
            # 在图像上绘制边界框和标签
            res.append((x1,y1,x2,y2,True))
            # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        else:
            # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            res.append((x1,y1,x2,y2,False))
    return res
        
def find_cars(image):
    modelyolo = YOLO("/Users/kumawu/code/FastSAM/weights/yolov8s.pt")
    # 使用 YOLO 模型检测汽车
    yolo_results = modelyolo(image)
    results = []
    for result in yolo_results:
        for box in result.boxes:
            # 获取类别索引和置信度
            cls_id = int(box.cls[0])  # 获取类别索引
            conf = float(box.conf[0])  # 获取置信度
            # 获取类别名称
            cls_name = result.names[cls_id]  # 从模型的类别名称列表中获取名称
            
            if cls_name == 'car':  # 检查是否为汽车类别
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取边界框坐标
                
                # 计算边界框的中心点
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center_point = (center_x, center_y)
                results.append((x1, y1, x2, y2, center_point))
    return results     
            
            
                
def find_road(params):

    
    
    
    
    results,final_image,binary_masks,red_overlays,combined_mask = detect(params)
    # show_combined_mask(combined_mask)
    # coords = get_mask_coordinates(combined_mask)
    # 对 combined_mask 进行形态学闭运算，消除噪点
    kernel = np.ones((50, 50), np.uint8) # 定义一个 50x50 的矩形结构元素
    processed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    
    # 找轮廓
    contours, hierarchy = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    print("轮廓数量：", len(contours))
    # 找到面积最大的轮廓
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)

        ratio = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
        for r in ratio:
            epsilon = r * cv2.arcLength(contour, True)

            approx = cv2.approxPolyDP(contour, epsilon, True)
            print("多边形顶点数量：", len(approx))
            if len(approx) == 4:
                
                # 获取顶点坐标
                vertices = approx.reshape(-1, 2)
                print("四个顶点坐标：", vertices)

                # # 在原始图像上绘制四边形
                # image = cv2.imread(params["image_path"])
                image = final_image.copy()  # 确保不修改原始图像
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)  # 绿色四边形
                # # 将 combined_mask 转换为红色通道的彩色图像
                # mask_colored = np.zeros_like(image)
                # mask_colored[:, :, 2] = combined_mask[:, :, 0]  # 将红色通道设置为 combined_mask

                # # 叠加 combined_mask 到原始图像上
                # overlay = cv2.addWeighted(image, 0.7, combined_mask, 0.3, 0)
                



                
                break
            else:
                print("无法找到四边形的顶点，请尝试调整 epsilon 值或检查 combined_mask 的形状。")
    else:
        print("未找到任何轮廓，请检查 combined_mask 的内容。")
    return approx,vertices
        

    
    



if __name__ == '__main__':
    params = {
        "image_path": "/Users/kumawu/code/FastSAM/images/abc.jpg",
        "save_dir": "./output",
        "prompts": "road"
    }
    image = cv2.imread(params["image_path"])
    approx,vertices = find_road(params)
    cars = find_cars(params["image_path"])
    results = is_point_in_polygon(cars, approx)
    for result in results:
        x1, y1, x2, y2, inside = result
        if inside:
            print(f"汽车在区域内：({x1}, {y1}) - ({x2}, {y2})")
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        else:
            print(f"汽车在区域外：({x1}, {y1}) - ({x2}, {y2})")
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # 显示结果
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    