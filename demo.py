import cva_gym # CVA GYM
import time
import numpy as np

env = cva_gym.make('simple_circuit') # env make
s = env.reset() # env reset


while True:
    linear_x = np.random.uniform(0, 0.5) # 랜덤으로 action 선택
    angular_z = np.random.uniform(-0.4, 0.4) # 랜덤으로 action 선택
    s, reward, done = env.step([linear_x, angular_z]) # env step
    print(reward)
    print(done)
    time.sleep(0.03)

    if done:
        print("Done!!!")
        s = env.reset()
        time.sleep(1)