import numpy as np

from envs.observation.decision_traffic_rules.feature_indices import agent_feat_id
from envs.observation.decision_traffic_rules.traffic_sign_db import traffic_feat_idx

class Preprocessor:
    def __init__(self):
        # Add your initialization logic here
        self.ego_attr_keep = np.array([agent_feat_id["x"], agent_feat_id["y"], agent_feat_id['yaw'],
                                       agent_feat_id["vx"], agent_feat_id["vy"],
                                       agent_feat_id["length"], agent_feat_id["width"]])
        self.neighbor_attr_keep = np.array([agent_feat_id["x"], agent_feat_id["y"], agent_feat_id['yaw'],
                                            agent_feat_id["vx"], agent_feat_id["vy"],
                                            agent_feat_id["length"], agent_feat_id["width"], 
                                            agent_feat_id["class"]])
        self.map_attr_keep = np.array([traffic_feat_idx["cl_x"], traffic_feat_idx["cl_y"], traffic_feat_idx["cl_yaw"], 
                                       traffic_feat_idx['ll_x'], traffic_feat_idx["ll_y"], traffic_feat_idx["ll_yaw"],
                                       traffic_feat_idx['ll_x'], traffic_feat_idx["ll_y"], traffic_feat_idx["ll_yaw"],
                                       traffic_feat_idx["speed_limit"]])
        self.FOV = 50
        self.max_speed = 80 / 3.6  # Convert km/h to m/s
        
    def preprocess_observation(self, observation):
        self.observations_before_preprocessing = observation
        # remove unnecessary attributes
        ego_data = observation.get('ego')[..., self.ego_attr_keep]
        neighbors_data = observation.get('neighbors')[..., self.neighbor_attr_keep]
        map_data = observation.get('map')[..., self.map_attr_keep]

        observation = {
            'ego': ego_data,
            'neighbors': neighbors_data,
            'map': map_data
        }

        observation = self.normalize(observation)

        return observation

    def preprocess_action(self, action):
        # Add your action preprocessing logic here
        pass

    def preprocess_reward(self, reward):
        # Add your reward preprocessing logic here
        pass

    def normalize(self, observation):
        for key in observation:
            # Normalize the dimensions of the observation
            if key == 'ego' or key == 'neighbors':
                observation[key][..., [0, 1, 5, 6]] /= self.FOV
                # Remove points outside the FOV
                observation[key][..., [0, 1, 5, 6]] = np.where(observation[key][..., [0, 1, 5, 6]] > self.FOV, 0, observation[key][..., [0, 1, 5, 6]])
                observation[key][..., 2] /= np.pi
                observation[key][..., [3, 4]] /= self.max_speed
                observation[key][..., [3, 4]] = (observation[key][..., [3, 4]] - 0.5) * 2
            elif key == 'map':
                observation[key][..., [0, 1, 3, 4, 6, 7]] /= self.FOV
                observation[key][..., 9] /= self.max_speed
                observation[key][..., 9] = (observation[key][..., 9] - 0.5) * 2
        return observation