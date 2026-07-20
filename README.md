
<img width="4096" height="2304" alt="quaddy" src="https://github.com/user-attachments/assets/e28867ae-8a8f-4742-a787-cdbe33900bd9" />
<img width="4096" height="2304" alt="quaddy" src="[https://github.com/user-attachments/assets/e28867ae-8a8f-4742-a787-cdbe33900bd9](https://youtu.be/XwfNStlKkAc)" />

# Reinforcement-learning-for-a-quadruped
Here, I trained a quadruped with 12 DOF and proximity sensor to walk towards a fixed direction. I used stablebaseline3 with PPO policy and Coppeliasim as the training environment.
Notes:
1. quaddy.ttt is the Coppeliasim scene.
2. quaddy_rl_script is the training script with on robot.
3. quaddy_rl_script_multi is the training script with 4 robots to cut training time.
4. ppo_quadruped_multi is the zip folder with trained model.
5. Used chatgpt for scripting and to comprehend what I was doing.
6. Some methods that you write are not explicitly called in your script but within the train method of stablebaseline3. Just for clarification.
7. To see the demonstration of the trained model, download all the files inside a folder and run the deploy_quaddy.py. Start Coppeliasim beforehand if necessary.

-- Start 4 windows of Coppeliasim for the four training environments if using multiple training environments and ensure the ZMQ server address match. Increase the number of training environment and Coppeliasim to match if your computer can handle to cut training time. It took around 7 hours in my laptop with 1.6 ghz CPU. I ran coppeliasim with the animation graphics turned off(the éye'button).
