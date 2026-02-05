from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes_mini', verbose=True)
# https://blog.csdn.net/qq_16137569/article/details/121066977
# 文件夹目录：
# ├── nuscenes
# │   ├── maps: 地图数据，四张地图对应着4个数据采集地点
# │   ├── samples: (带有标注信息的)关键帧数据，训练主要用这部分数据
# │   ├── sweeps: 完整时序数据，不带有标注信息，一般用于跟踪任务
# │   ├── v1.0-mini: 存有数据依赖关系、标注信息、标定参数的各种json文件
# │   ├── nuscenes_infos_temporal_train_mono3d.coco.json: coco 风格的 .json 文件更适合基于图像的方法，例如基于图像的 2D 和 3D 检测。
# │   ├── nuscenes_infos_temporal_val_mono3d.coco.json
# │   ├── nuscenes_infos_temporal_train.pkl: 训练数据集信息，每帧信息有两个键值： metadata 和 infos。 metadata 包含数据集本身的基本信息，例如 {'version': 'v1.0-trainval'}，而 infos 包含详细信息如下：https://blog.csdn.net/m0_45388819/article/details/121212307?ops_request_misc=&request_id=&biz_id=102&utm_term=Nuscenes数据集的pkl文件有啥&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-7-121212307.nonecase&spm=1018.2226.3001.4187
# │   ├── nuscenes_infos_temporal_val.pkl: .pkl 文件一般用于涉及点云的方法;
'''
数据集信息总体情况：
nuScenes数据集分为mini、trainval、test三个部分，每个部分的数据结构完全相同，可以分成scene、sample、sample_data三个层级，
数据访问通过token（可以理解为指针）来实现：


                                                                        camera
v1.0-mini                                               sensor data  {  lidar  : [file_path, time_stamp, ego_token, calib_token]
v1.0-trainval -->  NuScenes --> scene --> sample --> {                  radar
v1.0-test                                               annotation : [category：类别名称，包含10个检测类别, 
                                                                      translation：3D框的中心位置(x,y,z=0)，单位m，是全局坐标系下的坐标, 
                                                                      rotation：3D框的旋转量，用四元数(w,x,y,z)表示 全局坐标系到自车坐标系的旋转角
                                                                      size：3D框的尺寸(w,l,h)，单位米]


nuScenes存在四个坐标系：全局坐标系、车身坐标系、相机坐标系和激光坐标系。后面三个比较好理解，都是相对坐标系，目标的位置随本车的运动而变化；
而全局坐标系是绝对坐标系，是目标在地图中的绝对坐标，不随本车的运动而变化。标注真值的坐标是全局坐标系下的坐标。
所有转换都必须先转到车身坐标系(ego vehicle frame)，然后再转换到目标坐标系。
1. 标注真值(global frame)转换到激光坐标系(lidar frame)：使用位姿补偿转换到车身坐标系，然后再根据激光雷达外参转换到激光坐标系。
2. 标注真值(global frame)投影到图像(pixel coord)：使用位姿补偿转换到车身坐标系，然后再根据相机外参转换到相机坐标系，最后使用相机内参得到像素坐标系下的坐标。标注真值到车身坐标系的过程和上面类似，不过calib_data和ego_data需要从camera_data中获取，得到标注3D框在相机坐标系下的角点坐标points后，然后再使用相机内参投影至图像。
3. 激光真值(lidar frame)投影至图像(pixel coord)就相对麻烦一点，因为图像和激光时间戳不一致，需要多进行一步时间戳的变换。
'''

# 1. 场景 scene（共10个）
# 每个sence是一段约20s的视频片段，关键帧采样频率为2Hz，所以每个scene大约包含40个关键帧，所以mini共有10*20*2=400（实际404）个sample.
# 可以通过scene中的pre和next来访问上下相邻的sample.
# Mini：从训练/验证集抽取10个场景组成，包含完整的原始数据和标注信息，主要用于数据集的熟悉；
# TrainVal：训练/验证集，包含850个场景，其中700个训练场景，150个验证场景
# Test：测试集，包含150个场景，不包含标注数据。

nusc.list_scenes()  # 10个场景的信息
nusc.scene[0]   # 10个场景之1，取出第一个20s场景的总体信息，如下：
'''
第一个scene场景的总体信息，如下：
{'token': 'cc8c0bf57f984915a77078b10eb33198',   # token是唯一标识    
 'log_token': '7e25a2c8ea1f41c5b0da1e69ecfa71a2',
 'nbr_samples': 39,
 'first_sample_token': 'ca9a282c9e77460f8360f564131a8af5', 该场景下，第一个sample的token，该场景有约20s*2hz个这样的sample token
 'last_sample_token': 'ed5fc18c31904f96a8f0dbb99ff069c0',
 'name': 'scene-0061',
 'description': 'Parked truck, construction, intersection, turn left, following a van'}
'''


# 2. 样本 sample （共404 = 10*20*2）sence相当于20s的视频，sample就是每0.5s取一帧的图像
# 存储该样本下的token信息，不含真实数据
# 对应着一个关键帧（样本）数据，存储了相机、激光雷达、毫米波雷达的token信息，mini和trainval数据集中的sample还存储了标注信息的token
first_scene = nusc.scene[0]
first_sample_token = first_scene['first_sample_token']  # 获取第一个场景的第一个sample的token值
first_sample = nusc.get('sample', first_sample_token)  # 通过token，取出第一个场景scene的第一个样本sample的token信息，如下：
'''
第一个场景scene的第一个样本sample的token信息，如下:
{'token': 'ca9a282c9e77460f8360f564131a8af5',                               第一个scene的第一个sample的token
 'timestamp': 1532402927647951,
 'prev': '',
 'next': '39586f9d59004284a7114a68825e8eec',
 'scene_token': 'cc8c0bf57f984915a77078b10eb33198',                         第一个scene的token
 'data': {'RADAR_FRONT': '37091c75b9704e0daa829ba56dfa0906', 雷达token信息
  'RADAR_FRONT_LEFT': '11946c1461d14016a322916157da3c7d',
  'RADAR_FRONT_RIGHT': '491209956ee3435a9ec173dad3aaf58b',
  'RADAR_BACK_LEFT': '312aa38d0e3e4f01b3124c523e6f9776',
  'RADAR_BACK_RIGHT': '07b30d5eb6104e79be58eadf94382bc1',
  'LIDAR_TOP': '9d9bf11fb0e144c8b446d54a8a00184f', 激光雷达token信息
  'CAM_FRONT': 'e3d495d4ac534d54b321f50006683844', 相机token信息
  'CAM_FRONT_RIGHT': 'aac7867ebf4f446395d29fbd60b63b3b',
  'CAM_BACK_RIGHT': '79dbb4460a6b40f49f9c150cb118247e',
  'CAM_BACK': '03bea5763f0f4722933508d5999c5fd8',
  'CAM_BACK_LEFT': '43893a033f9c46d4a51b5e08a67a1eb7',
  'CAM_FRONT_LEFT': 'fe5422747a7d4268a4b07fc396707b23'},
 'anns': ['ef63a697930c4b20a6b9791f423351da',                               第一个sample有66个框，标注的token信息
  '6b89da9bf1f84fd6a5fbe1c3b236f809',
    ......此处省略66个anns
  '2bfcc693ae9946daba1d9f2724478fd4']}
'''

# 3. 样本数据 sample data
# sample中存储的token指向的数据，即我们最终真正关心的信息，比如图片路径、位姿数据、传感器标定结果、标注目标的3d信息等。获取到这些信息就可以开始训练模型了
cam_front_data = nusc.get('sample_data',first_sample['data']['RADAR_FRONT']) # 获取第一个场景、第一个sample的前摄像机的信息，如下：

'''
第一个场景、第一个sample的前摄像头CAM_FRONT的信息，因为有404个sample，所以一共有404张前摄像头的图片，如下：
{'token': 'e3d495d4ac534d54b321f50006683844',                           相机token
 'sample_token': 'ca9a282c9e77460f8360f564131a8af5',                    该相机所属的sample的token
 'ego_pose_token': 'e3d495d4ac534d54b321f50006683844',
 'calibrated_sensor_token': '1d31c729b073425e8e0202c5c6e66ee1',
 'timestamp': 1532402927612460,
 'fileformat': 'jpg',
 'is_key_frame': True,
 'height': 900,
 'width': 1600,
 'filename': 'samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg',
 'prev': '',
 'next': '68e8e98cf7b0487baa139df808641db7',
 'sensor_modality': 'camera',
 'channel': 'CAM_FRONT'}
'''
# nusc.render_sample_data(cam_front_data['token'])    # 可视化相机图

# 4. 样本标注 sample annotation （共18538个，=404个sample的每个sample里，1+5+6个传感器都各自标注信息；一个sample里，即一张图里会有很多个标注框，例如行人、街边障碍水马等非常多）
first_annotation_token = first_sample['anns'][0] # 第一个scene的，第一个sample的, 第一个annotation = 'ef63a697930c4b20a6b9791f423351da'
first_annotation_metadata = nusc.get('sample_annotation',first_annotation_token)
'''
第一个scene的，第一个sample的, 第一个annotation的信息，如下：
{'token': 'ef63a697930c4b20a6b9791f423351da',
 'sample_token': 'ca9a282c9e77460f8360f564131a8af5',
 'instance_token': '6dd2cbf4c24b4caeb625035869bca7b5',
 'visibility_token': '1',
 'attribute_tokens': ['4d8821270b4a47e3a8a300cbec48188e'],
 'translation': [373.256, 1130.419, 0.8], 位置
 'size': [0.621, 0.669, 1.642], 尺寸
 'rotation': [0.9831098797903927, 0.0, 0.0, -0.18301629506281616], 方向
 'prev': '',
 'next': '7987617983634b119e383d8a29607fd7',
 'num_lidar_pts': 1,
 'num_radar_pts': 0,
 'category_name': 'human.pedestrian.adult'} 类别
'''
# nusc.render_annotation(first_annotation_metadata['token'])   # 可视化标注信息

'''
nuscenes_infos_temporal_val_mono3d.coco.json如下：
{"annotations": [
  {"file_name": "samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg", 
   "image_id": "4f5e35aa6c6a426ca945e206fb2f4921", 
   "area": 1887.8372447879883, 
   "category_name": "pedestrian", 
   "category_id": 7, 
   "bbox": [553.4947253760256, 490.5584879921071, 30.329964533936845, 62.24330538453637], 
   "iscrowd": 0, 
   "bbox_cam3d": [-7.516707974170363, 1.5012318792386194, 36.525215135341675, 0.647, 1.778, 0.621, 1.7942699713480013], 
   "velo_cam3d": [-0.18470906572029236, -1.0365737419422854], 
   "center2d": [568.7653969700092, 521.4768430424537, 36.525215135341675], 
   "attribute_name": "pedestrian.moving", 
   "attribute_id": 2, 
   "segmentation": [], "id": 0}, 
'''



# 5. 实例 instance
my_instance = nusc.instance[0]
instance_token = my_instance['token']
# nusc.render_instance(instance_token)  # 可视化

# 6. 类别 category
nusc.list_categories()

# 7. 根据sample_token，找出CAM_FRONT的图片
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=False)
sample_token = '55c258972acb4300a3a6077a531ab050'
print(nusc.get('sample_data',nusc.get('sample', sample_token)['data']['CAM_BACK'])['filename'])

# 8. map信息
nusc.map[0]
'''
{'category': 'semantic_prior',
 'token': '53992ee3023e5494b90c316c183be829',               # map_token
 'filename': 'maps/53992ee3023e5494b90c316c183be829.png',
 'log_tokens': ['0986cb758b1d43fdaa051ab23d45582b',         # 该地图所包含的log_token
  '1c9b302455ff44a9a290c372b31aa3ce',
  'e60234ec7c324789ac7c8441a5e49731',
  '46123a03f41e4657adc82ed9ddbe0ba2',
'''

# 9. log信息
nusc.log[0]
'''
'token': '7e25a2c8ea1f41c5b0da1e69ecfa71a2',        # log_token。scene信息里包含了log_token，即告诉人们采集该场景的背景信息
  'logfile': 'n015-2018-07-24-11-22-45+0800',   
  'vehicle': 'n015',                                # 采集车辆
  'date_captured': '2018-07-24',                    # 采集日期
  'location': 'singapore-onenorth',                 # 地图位置
  'map_token': '53992ee3023e5494b90c316c183be829'}  # 地图文件名
'''

# 10. 世界坐标系 -> 自车坐标系 -> Lidar坐标系
import numpy as np
from pyquaternion import Quaternion

sample_token = '858a1ece22cf45d9bc71e42336604b78'
sample_rec = nusc.get('sample', sample_token)                           # 获取该sample_token所对应的sample的所有真实信息
sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])    # 获取该sample的Lidar信息
calib_data = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])

pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])         # 获取该sample的ego信息

ann = nusc.get('sample_annotation','ed4cfcdd5e5b4a5b9af17a0f3ec79482')  # gt框的标注信息
center = np.array(ann['translation'])                                   # gt的世界坐标
orientation = np.array(ann['rotation'])                                 # gt的方向


# 世界->自车
# 相当于： Pw = ego2world*Pe， Pe = (ego2world).inv * Pw
quaternion = Quaternion(pose_record['rotation']).inverse
center -= np.array(pose_record['translation'])
center = np.dot(quaternion.rotation_matrix, center)

quaternion = Quaternion(calib_data['rotation']).inverse
center -= np.array(calib_data['translation'])
center = np.dot(quaternion.rotation_matrix, center)
print(center)

# 11. 根据sample_token获取后视相机图片
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=False)
sample_token = 'ff64ab8d8f6c4af4920aea853f437203'
print(nusc.get('sample_data',nusc.get('sample', sample_token)['data']['CAM_FRONT'])['filename'])


# 12. 可视化样本(图片) 'b5989651183643369174912bc5641d3b'，'0d0700a2284e477db876c3ee1d864668'
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=False)
nusc.render_sample('e7a4e9440654403ebc21f1ac13924cf1')

# 13. 可视化单个目标
nusc.render_annotation('218a5ea3e1b04aa6b87725f72f1fbc12')

# 14. 获得单个ann的BBOX信息
nusc.get('sample_annotation','eebc9b889aaf426ab3b458d735f4efc0')

# 15. 根据sample_token 获得ann
nusc.get('sample', '06be0e3b665c44fa8d17d9f4770bdf9c')