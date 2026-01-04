import yaml
import argparse
from easydict import EasyDict as ed


def get_args():
    parser = argparse.ArgumentParser(description="training vae configure")
    parser.add_argument("--config", help="Base configure file name", type=str, default='configs/test.yaml')
    parser.add_argument("--setting", help="setting1 is no consideration of physical simulation, no consideration of robotic arms\
                                                         setting2 is consideration of physical simulation, no consideration of robotic arms\
                                                         setting3 is consideration of physical simulation, consideration of robotic arms", type=int, default=1)
    parser.add_argument("--is_3d_packing", help="3d packing or 2d packing, 1 or 0", type=int, default=1)
    parser.add_argument("--data", help="regular data: random, time_series, occupancy, flat_long.\
                                irregular data: blockout, armbench, abc. 2d data: steel_plate, panel_furniture", type=str, default='occupancy')
    parser.add_argument("--method", help="regular method: LeftBottom, HeightmapMin, LSAH, MACS, RANDOM, OnlineBPH, DBL, BR, SDFPack, PCT, PackE. \
                                irregular method: SDF_Pack, MTPE, IR_HM, BLBF, FF, IR_BPP. 2d method: pack_2d", type=str, default='SDFPack')
    parser.add_argument("--config_learning_method", help="Test learning method configure file name: pct.yaml, packe.yaml", type=str, default='configs/pct.yaml')
    parser.add_argument("--test_data_config", help="test data configuration, 0 is the default data, the range is 0-29", type=int, default=2)

    args = parser.parse_args()
    args_base = load_config(args.config)
    
    args_base = get_data_info(args, args_base)

    return args, args_base


def load_config(config_dir, easy=True):
    cfg = yaml.load(open(config_dir), yaml.FullLoader)
    if easy is True:
        cfg = ed(cfg)
    return cfg


def get_data_info(args, args_base):
    # regular data: random, time_series, occupancy, flat_long
    # irregular data: blockout, armbench, abc
    # 2d data: steel_plate, panel_furniture

    if args.data in ['blockout', 'armbench', 'abc']:
        # if data is 3d irregular, we use UR5
        args_base.Scene.block_unit = args_base.Scene.UR5.block_unit_ur5
        args_base.Scene.arm_urdf_path = args_base.Scene.UR5.arm_urdf_path_ur5
        args_base.Scene.homej = args_base.Scene.UR5.homej_ur5
    else:
        # args_base.Scene.block_unit = args_base.Scene.IRB2400.block_unit_irb2400
        # args_base.Scene.arm_urdf_path = args_base.Scene.IRB2400.arm_urdf_path_irb2400
        # args_base.Scene.homej = args_base.Scene.IRB2400.homej_irb2400
        args_base.Scene.block_unit = args_base.Scene.IRB6700.block_unit_irb6700
        args_base.Scene.arm_urdf_path = args_base.Scene.IRB6700.arm_urdf_path_irb6700
        args_base.Scene.homej = args_base.Scene.IRB6700.homej_irb6700


    # 3d regular
    if args.data == 'random':
        container_size = args_base.Data.Random.container_size
        easy_container_size = [container_size[0], container_size[1], min(container_size[0], container_size[1])]
        args_base.Scene.target_container_size = easy_container_size
        args_base.Scene.target_origin = args_base.Data.Random.target_origin
        args_base.Scene.threshold = args_base.Data.Random.threshold
        
        args_base.Scene.ompl_plan_allowed_time = args_base.Data.Random.ompl_plan_allowed_time
        args_base.Scene.interpolate_num = args_base.Data.Random.interpolate_num
        args_base.Scene.step_size = args_base.Data.Random.step_size
    elif args.data == 'time_series':
        args_base.Scene.target_container_size = args_base.Data.Time_series.container_size
        args_base.Scene.target_origin = args_base.Data.Time_series.target_origin
        args_base.Scene.threshold = args_base.Data.Time_series.threshold

        args_base.Scene.ompl_plan_allowed_time = args_base.Data.Time_series.ompl_plan_allowed_time
        args_base.Scene.interpolate_num = args_base.Data.Time_series.interpolate_num
        args_base.Scene.step_size = args_base.Data.Time_series.step_size
    elif args.data == 'occupancy':
        args_base.Scene.target_container_size = args_base.Data.Occupancy.container_size
        args_base.Scene.target_origin = args_base.Data.Occupancy.target_origin
        args_base.Scene.threshold = args_base.Data.Occupancy.threshold

        args_base.Scene.ompl_plan_allowed_time = args_base.Data.Occupancy.ompl_plan_allowed_time
        args_base.Scene.interpolate_num = args_base.Data.Occupancy.interpolate_num
        args_base.Scene.step_size = args_base.Data.Occupancy.step_size
    elif args.data == 'flat_long':
        args_base.Scene.target_container_size = args_base.Data.Flat_long.container_size
        args_base.Scene.target_origin = args_base.Data.Flat_long.target_origin
        args_base.Scene.threshold = args_base.Data.Flat_long.threshold

        args_base.Scene.ompl_plan_allowed_time = args_base.Data.Flat_long.ompl_plan_allowed_time
        args_base.Scene.interpolate_num = args_base.Data.Flat_long.interpolate_num
        args_base.Scene.step_size = args_base.Data.Flat_long.step_size

    # 3d irregular
    elif args.data == 'blockout':
        container_size = args_base.Data.Blockout.container_size
        easy_container_size = [container_size[0], container_size[1], min(container_size[0], container_size[1])]
        args_base.Scene.target_container_size = easy_container_size
        args_base.Scene.target_origin = args_base.Data.Blockout.target_origin
        args_base.Scene.objPath = args_base.Data.Blockout.objPath
        args_base.Scene.scale = args_base.Data.Blockout.scale
        
        args_base.Scene.ompl_plan_allowed_time = args_base.Data.Blockout.ompl_plan_allowed_time
        args_base.Scene.interpolate_num = args_base.Data.Blockout.interpolate_num
        args_base.Scene.step_size = args_base.Data.Blockout.step_size
    elif args.data == 'armbench':
        container_size = args_base.Data.Armbench.container_size
        easy_container_size = [container_size[0], container_size[1], min(container_size[0], container_size[1])]
        args_base.Scene.target_container_size = easy_container_size
        args_base.Scene.target_origin = args_base.Data.Armbench.target_origin
        args_base.Scene.objPath = args_base.Data.Armbench.objPath
        args_base.Scene.scale = args_base.Data.Armbench.scale

        args_base.Scene.ompl_plan_allowed_time = args_base.Data.Armbench.ompl_plan_allowed_time
        args_base.Scene.interpolate_num = args_base.Data.Armbench.interpolate_num
        args_base.Scene.step_size = args_base.Data.Armbench.step_size
    elif args.data == 'abc':
        container_size = args_base.Data.Abc.container_size
        easy_container_size = [container_size[0], container_size[1], min(container_size[0], container_size[1])]
        args_base.Scene.target_container_size = easy_container_size
        args_base.Scene.target_origin = args_base.Data.Abc.target_origin
        args_base.Scene.objPath = args_base.Data.Abc.objPath
        args_base.Scene.scale = args_base.Data.Abc.scale

        args_base.Scene.ompl_plan_allowed_time = args_base.Data.Abc.ompl_plan_allowed_time
        args_base.Scene.interpolate_num = args_base.Data.Abc.interpolate_num
        args_base.Scene.step_size = args_base.Data.Abc.step_size

    # 2d
    elif args.data == 'steel_plate':
        container_size = args_base.Data.Steel_plate.container_size
        easy_container_size = [container_size[0], container_size[1], min(container_size[0], container_size[1])]
        args_base.Scene.target_container_size = easy_container_size
        args_base.Scene.target_origin = args_base.Data.Steel_plate.target_origin
        args_base.Scene.objPath_2d = args_base.Data.Steel_plate.objPath_2d
        args_base.Scene.scale_2d = args_base.Data.Steel_plate.scale_2d
        # args_base.Scene.img_path_2d = args_base.Data.Steel_plate.img_path_2d

        args_base.Scene.ompl_plan_allowed_time = args_base.Data.Steel_plate.ompl_plan_allowed_time
        args_base.Scene.interpolate_num = args_base.Data.Steel_plate.interpolate_num
        args_base.Scene.step_size = args_base.Data.Steel_plate.step_size
    elif args.data == 'panel_furniture':
        container_size = args_base.Data.Panel_furniture.container_size
        easy_container_size = [container_size[0], container_size[1], min(container_size[0], container_size[1])]
        args_base.Scene.target_container_size = easy_container_size
        args_base.Scene.target_origin = args_base.Data.Panel_furniture.target_origin
        args_base.Scene.objPath_2d = args_base.Data.Panel_furniture.objPath_2d
        args_base.Scene.scale_2d = args_base.Data.Panel_furniture.scale_2d
        # args_base.Scene.img_path_2d = args_base.Data.Panel_furniture.img_path_2d

        args_base.Scene.ompl_plan_allowed_time = args_base.Data.Panel_furniture.ompl_plan_allowed_time
        args_base.Scene.interpolate_num = args_base.Data.Panel_furniture.interpolate_num
        args_base.Scene.step_size = args_base.Data.Panel_furniture.step_size


    return args_base