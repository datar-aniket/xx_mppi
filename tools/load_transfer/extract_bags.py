#!/usr/bin/env python3
"""Extract the signals the load-transfer analysis needs from rosbag2 (mcap) runs.

Writes one <bag name>.npz per bag with two arrays:
  ekf  /ekf/state rows, columns listed in lt_data.EKF_COLUMNS
  dc   /direct_control rows: bag time, stamp, front steer, rear steer, throttle, type

Needs a sourced workspace so xxcar_msgs can be deserialized:
  source ~/fireball_ws/install/setup.bash
  ./extract_bags.py ~/bags/iros* --out /tmp/load_transfer
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def stamp_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def extract(bag_dir, out_dir):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    storage_id = 'mcap' if any(f.endswith('.mcap') for f in os.listdir(bag_dir)) else 'sqlite3'
    reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id=storage_id),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    topics = [t for t in ('/ekf/state', '/direct_control') if t in types]
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    classes = {t: get_message(types[t]) for t in topics}
    ekf, dc = [], []
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        m = deserialize_message(data, classes[topic])
        t_bag = t_ns * 1e-9
        if topic == '/ekf/state':
            q = m.pose.orientation
            ekf.append([
                t_bag, stamp_s(m.header.stamp),
                m.pose.position.x, m.pose.position.y, m.pose.position.z, q.x, q.y, q.z, q.w,
                m.twist.linear.x, m.twist.linear.y, m.twist.linear.z,
                m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z,
                m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z,
                m.side_slip_rad, m.motor_current_a, m.steering_angle, m.motor_speed_erpm,
                m.steering_angle_rear, m.solution_status, m.reset_counter, m.source_valid,
                float(m.rc_armed), float(m.rc_auto), float(m.rc_trigger_high),
                m.wheel_torque_nm, m.wheel_speed_mps])
        else:
            dc.append([t_bag, stamp_s(m.stamp), m.steering_angle_rad, m.rear_steering_angle_rad,
                       m.throttle, m.throttle_type])
    name = os.path.basename(os.path.normpath(bag_dir))
    out_path = os.path.join(out_dir, f'{name}.npz')
    np.savez(out_path, ekf=np.array(ekf), dc=np.array(dc))
    return f'{name}: {len(ekf)} ekf states, {len(dc)} commands -> {out_path}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bags', nargs='+', help='rosbag2 directories')
    parser.add_argument('--out', required=True, help='output directory for the .npz files')
    parser.add_argument('--jobs', type=int, default=os.cpu_count())
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for line in pool.map(extract, args.bags, [args.out] * len(args.bags)):
            print(line)
    return 0


if __name__ == '__main__':
    sys.exit(main())
