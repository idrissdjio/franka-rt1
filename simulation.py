import tensorflow as tf
import numpy as np
import pybullet as p
import pybullet_data
from tf_agents.policies import py_tf_eager_policy
import tf_agents
from tf_agents.trajectories import time_step as ts

# Initialize PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,0]
startOrientation = p.getQuaternionFromEuler([0,0,0])
robotId = p.loadURDF("franka_panda/panda.urdf", startPos, startOrientation)

# Load the TensorFlow Agents policy
saved_model_path = 'rt_1_x_tf_trained_for_002272480_step'
tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
    model_path=saved_model_path,
    load_specs_from_pbtxt=True,
    use_tf_function=True)

# Get the initial state of the policy
policy_state = tfa_policy.get_initial_state(batch_size=1)

# Simulate and apply actions from the policy
for i in range(1000):
    # Here you should generate the appropriate observation from the simulation
    # For now, we use a dummy observation. Replace this with real sensor data or simulation state
    observation = tf_agents.specs.zero_spec_nest(tf_agents.specs.from_spec(tfa_policy.time_step_spec.observation))

    # Generate a time_step object from the observation
    tfa_time_step = ts.transition(observation, reward=np.zeros((), dtype=np.float32))

    # Get the action from the policy
    policy_step = tfa_policy.action(tfa_time_step, policy_state)
    action = policy_step.action
    policy_state = policy_step.state

    # TODO: Convert 'action' to PyBullet joint control commands
    # This depends on the action space of your trained model
    # For example, if your action is a joint torque or position, you can apply it like this:
    # p.setJointMotorControl2(robotId, jointIndex, p.POSITION_CONTROL, targetPosition=action_value)

    p.stepSimulation()

p.disconnect()





